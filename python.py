import streamlit as st
import numpy as np
import pandas as pd
import io
import re
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor, Descriptors
import base64
from typing import Dict, List, Optional
from datetime import datetime

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Peptide Analyzer Pro", layout="wide", initial_sidebar_state="collapsed")

# --- СКРЫТИЕ ЭЛЕМЕНТОВ ИНТЕРФЕЙСА ---
hide_style = """
    <style>
    #GithubIcon {visibility: hidden;}
    div[data-testid="stStatusWidget"] {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .main .block-container {padding-top: 1rem;}
    .stChatMessage {background-color: #f8f9fa; border-radius: 15px; padding: 10px; margin: 5px 0;}
    .stButton button {border-radius: 20px;}
    .dev-status {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0; text-align: center;}
    .reaction-calc {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; margin: 10px 0;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# --- АВАТАРКИ ---
def get_user_avatar(): return "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
def get_bot_avatar(): return "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"

# --- БАЗА АТОМНЫХ МАСС ---
ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.012, 'B': 10.810,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
    'S': 32.060, 'Cl': 35.450, 'Ar': 39.950, 'K': 39.098, 'Ca': 40.078,
    'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.380, 'Ag': 107.868, 'Ba': 137.327,
    'Pb': 207.200, 'Au': 196.967, 'Hg': 200.592
}

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С МОЛЕКУЛАМИ ---
def get_molecule_info(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return {
                'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
                'mass': Descriptors.ExactMolWt(mol),
                'atoms': mol.GetNumAtoms(),
                'bonds': mol.GetNumBonds()
            }
    except:
        pass
    return None

def display_molecule(smiles: str, title: str):
    """Отображает структуру молекулы, если SMILES корректен"""
    if not smiles:
        st.warning(f"{title}: SMILES не указан")
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error(f"❌ {title}: Не удалось распознать SMILES: {smiles}")
        st.info("💡 Примеры правильных SMILES: CCO (этанол), CC(=O)O (уксусная кислота), CC(=O)OCC (этилацетат)")
        return False
    info = get_molecule_info(smiles)
    if not info:
        st.error(f"❌ {title}: Не удалось получить информацию о молекуле")
        return False
    with st.container():
        st.markdown(f"**{title}**")
        col1, col2 = st.columns([1, 1])
        with col1:
            rdDepictor.Compute2DCoords(mol)
            img = Chem.Draw.MolToImage(mol, size=(350, 250))
            st.image(img, use_container_width=True)
        with col2:
            st.markdown(f"""
            - **Формула:** {info['formula']}
            - **Молярная масса:** {info['mass']:.4f} г/моль
            - **Атомов:** {info['atoms']}
            - **Связей:** {info['bonds']}
            """)
    return True

# --- ФУНКЦИИ РАСЧЁТА МОЛЯРНЫХ МАСС ---
def calculate_inorganic_molar_mass(formula: str) -> Optional[float]:
    try:
        while '(' in formula:
            pattern = r'\(([^()]+)\)(\d*)'
            match = re.search(pattern, formula)
            if match:
                inner = match.group(1)
                count = int(match.group(2)) if match.group(2) else 1
                inner_mass = calculate_inorganic_molar_mass(inner)
                if inner_mass:
                    formula = formula.replace(f"({inner}){match.group(2)}", f"X{inner_mass * count}")
        pattern = r'([A-Z][a-z]*)(\d*)'
        matches = re.findall(pattern, formula)
        total = 0.0
        for el, cnt in matches:
            if el == 'X':
                total += float(cnt)
            elif el in ATOMIC_MASSES:
                cnt = int(cnt) if cnt else 1
                total += ATOMIC_MASSES[el] * cnt
            else:
                return None
        return total
    except:
        return None

def parse_hydrate(formula: str) -> Optional[float]:
    try:
        m = re.search(r'([^·.*]+)[·.*](\d+)H2O', formula)
        if m:
            main = m.group(1)
            water = int(m.group(2))
            main_mass = calculate_inorganic_molar_mass(main)
            if main_mass:
                return main_mass + 18.015 * water
    except:
        pass
    return None

def calculate_molar_mass(formula: str) -> Dict:
    res = {'molar_mass': None, 'formula_display': formula, 'error': None}
    if not formula:
        res['error'] = 'Формула не указана'
        return res
    if formula.startswith('smiles:'):
        smiles = formula[7:]
        info = get_molecule_info(smiles)
        if info:
            res['molar_mass'] = info['mass']
            res['formula_display'] = info['formula']
            return res
        else:
            res['error'] = 'Ошибка парсинга SMILES'
            return res
    if '·' in formula or '.' in formula or '*' in formula:
        mm = parse_hydrate(formula)
        if mm:
            res['molar_mass'] = mm
            return res
    mm = calculate_inorganic_molar_mass(formula)
    if mm:
        res['molar_mass'] = mm
    else:
        res['error'] = 'Не удалось распознать формулу'
    return res

# --- ФУНКЦИИ АНАЛИЗА ПЕПТИДОВ ---
def calculate_dihedral(p1, p2, p3, p4):
    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return round(float(np.degrees(np.arctan2(y, x))), 1)

def parse_mol2_and_get_mol(file_content):
    mol = Chem.MolFromMol2Block(file_content, removeHs=False)
    conformers, current_coords, heavy_atom_names = [], {}, []
    lines = file_content.splitlines()
    is_atom = False
    for line in lines:
        if line.startswith('@<TRIPOS>MOLECULE'):
            if current_coords:
                conformers.append(current_coords)
            current_coords, is_atom = {}, False
        elif line.startswith('@<TRIPOS>ATOM'):
            is_atom = True
        elif line.startswith(('@<TRIPOS>BOND', '@<TRIPOS>SUB')):
            is_atom = False
        elif is_atom:
            p = line.split()
            if len(p) >= 6:
                name = p[1]
                current_coords[name] = np.array([float(p[2]), float(p[3]), float(p[4])])
                if name not in heavy_atom_names and not name.upper().startswith('H'):
                    heavy_atom_names.append(name)
    if current_coords:
        conformers.append(current_coords)
    return mol, conformers, heavy_atom_names

def render_static_svg(mol, heavy_atom_names, bond_len, font_size, pan_x, pan_y, zoom):
    if not mol:
        return "<div>Ошибка: молекула не загружена</div>"
    try:
        display_mol = Chem.RemoveHs(mol)
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(display_mol)
        d2d = rdMolDraw2D.MolDraw2DSVG(1000, 500)
        opts = d2d.drawOptions()
        opts.fixedBondLength = bond_len
        opts.minFontSize = font_size
        opts.annotationFontScale = 0.8
        for i, atom in enumerate(display_mol.GetAtoms()):
            if i < len(heavy_atom_names):
                atom.SetProp("atomNote", heavy_atom_names[i])
        d2d.DrawMolecule(display_mol)
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()
        transform = f"transform='translate({pan_x}, {pan_y}) scale({zoom})'"
        svg = svg.replace('<g>', f'<g {transform}>', 1)
        svg = re.sub(r'width=[\'"].*?[\'"]', 'width="100%"', svg)
        svg = re.sub(r'height=[\'"].*?[\'"]', 'height="100%"', svg)
        return f"<div style='background:white; border-radius:12px; border:1px solid #ddd; overflow:auto;'>{svg}</div>"
    except Exception as e:
        return f"<div style='color:red; padding:20px;'>Ошибка отрисовки: {str(e)}</div>"

def calculate_angles_for_node(conformers, phi_atoms, psi_atoms):
    phi_vals, psi_vals = [], []
    for conf in conformers:
        try:
            phi = calculate_dihedral(conf[phi_atoms[0]], conf[phi_atoms[1]], conf[phi_atoms[2]], conf[phi_atoms[3]])
            psi = calculate_dihedral(conf[psi_atoms[0]], conf[psi_atoms[1]], conf[psi_atoms[2]], conf[psi_atoms[3]])
            phi_vals.append(phi)
            psi_vals.append(psi)
        except:
            phi_vals.append(0.0)
            psi_vals.append(0.0)
    return phi_vals, psi_vals

# --- КЛАСС ДЛЯ РАСЧЁТА РЕАКЦИЙ ---
class ReactionCalculator:
    def __init__(self):
        self.reagents = []
        self.products = []
    def add_reagent(self, name, formula, coefficient, mass, equivalents):
        calc = calculate_molar_mass(formula)
        self.reagents.append({'name': name, 'formula': formula, 'coefficient': coefficient,
                              'mass': mass, 'equivalents': equivalents, 'calc': calc})
    def add_product(self, name, formula, coefficient):
        calc = calculate_molar_mass(formula)
        self.products.append({'name': name, 'formula': formula, 'coefficient': coefficient, 'calc': calc})
    def calculate(self, use_equivalents):
        results = []
        for r in self.reagents:
            calc = r['calc']
            if calc['molar_mass'] and r['mass'] > 0:
                moles = r['mass'] / calc['molar_mass']
                if use_equivalents:
                    adj = moles * r['equivalents'] / r['coefficient']
                else:
                    adj = moles / r['coefficient']
                results.append({'name': r['name'], 'formula_display': calc['formula_display'],
                                'coefficient': r['coefficient'], 'mass': r['mass'],
                                'molar_mass': calc['molar_mass'], 'moles': moles, 'adj_moles': adj,
                                'equivalents': r['equivalents'] if use_equivalents else 1})
        if results:
            limiting = min(results, key=lambda x: x['adj_moles'])
            for r in results:
                if r['name'] == limiting['name']:
                    r['limiting'] = True
        return results
    def calculate_products(self, results, use_equivalents):
        limiting = next((r for r in results if r.get('limiting')), None)
        if not limiting:
            return []
        out = []
        for p in self.products:
            calc = p['calc']
            if calc['molar_mass']:
                moles = (limiting['adj_moles'] / limiting['coefficient']) * p['coefficient']
                mass = moles * calc['molar_mass']
                out.append({'name': p['name'], 'formula_display': calc['formula_display'],
                            'coefficient': p['coefficient'], 'molar_mass': calc['molar_mass'],
                            'moles': moles, 'mass': mass})
        return out
    def get_reaction_data(self):
        return {'reagents': self.reagents, 'products': self.products}

# --- ИНИЦИАЛИЗАЦИЯ СЕССИИ ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "🧪 Добро пожаловать!"}]
if "saved_reaction" not in st.session_state:
    st.session_state.saved_reaction = None
if "show_product_structure" not in st.session_state:
    st.session_state.show_product_structure = {}

# --- ИНТЕРФЕЙС ---
st.title("🧪 Peptide Analyzer Pro")
chat_col, main_col = st.columns([1, 2])

with chat_col:
    st.markdown("### 💬 Чат")
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.messages[-10:]:
            with st.chat_message(msg["role"], avatar=get_bot_avatar() if msg["role"]=="assistant" else get_user_avatar()):
                st.markdown(msg["content"])
    if prompt := st.chat_input("Введите сообщение..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": "Используйте вкладки справа."})
        st.rerun()

with main_col:
    tab1, tab2, tab3 = st.tabs(["🔬 Анализ пептидов", "🧪 Расчет реакций", "⚖️ Расчет по массе"])

    # --- АНАЛИЗ ПЕПТИДОВ ---
    with tab1:
        st.markdown('<div class="reaction-calc">🔬 Анализ пептидов</div>', unsafe_allow_html=True)
        with st.expander("⚙️ Настройки"):
            b_len = st.slider("Длина связей", 20, 150, 60)
            f_size = st.slider("Размер шрифта", 6, 30, 14)
            zoom = st.slider("Масштаб", 0.5, 2.0, 1.0)
        uploaded = st.file_uploader("Загрузите .mol2", type=['mol2'])
        if uploaded:
            content = uploaded.read().decode("utf-8")
            mol, confs, atoms = parse_mol2_and_get_mol(content)
            if mol and confs:
                st.success(f"Загружено {len(confs)} конформеров, {len(atoms)} атомов")
                svg = render_static_svg(mol, atoms, b_len, f_size, 0, 0, zoom)
                st.components.v1.html(svg, height=500)
                st.subheader("Выбор атомов для анализа")
                num_nodes = st.number_input("Количество узлов", 1, 3, 1)
                for i in range(num_nodes):
                    st.markdown(f"**Узел {i+1}**")
                    cols = st.columns(2)
                    with cols[0]:
                        phi = st.multiselect(f"Phi атомы", atoms, max_selections=4, key=f"phi_{i}")
                    with cols[1]:
                        psi = st.multiselect(f"Psi атомы", atoms, max_selections=4, key=f"psi_{i}")
                    if len(phi)==4 and len(psi)==4:
                        phi_vals, psi_vals = calculate_angles_for_node(confs, phi, psi)
                        df = pd.DataFrame({"Конформер": range(1, len(phi_vals)+1), "Phi": phi_vals, "Psi": psi_vals})
                        st.dataframe(df, use_container_width=True)
                        fig = px.scatter(df, x="Phi", y="Psi", text="Конформер")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

    # --- РАСЧЁТ РЕАКЦИЙ ---
    with tab2:
        st.markdown('<div class="reaction-calc">🧪 Калькулятор реакций</div>', unsafe_allow_html=True)
        use_equiv = st.radio("Учитывать эквиваленты?", ["Да", "Нет"], horizontal=True)
        use_equiv_bool = (use_equiv == "Да")
        reaction_type = st.radio("Тип реакции:", ["🧬 Органическое + Органическое", "⚗️ Органическое + Неорганическое", "🧪 Неорганическое + Неорганическое"], horizontal=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Реагент 1**")
            if "Органическое" in reaction_type:
                sm1 = st.text_input("SMILES реагента 1:", key="r1_sm", placeholder="CCO, CC(=O)O")
                r1_f = f"smiles:{sm1}" if sm1 else ""
                if sm1 and st.button("🔍 Показать структуру 1", key="show_r1"):
                    display_molecule(sm1, "Реагент 1")
            else:
                r1_f = st.text_input("Формула реагента 1:", key="r1_f", placeholder="NaCl, H2SO4")
            coeff1 = st.number_input("Коэффициент 1:", 0.1, 10.0, 1.0, key="c1")
            mass1 = st.number_input("Масса 1 (г):", 0.0, 1000.0, 0.0, key="m1")
            equiv1 = st.number_input("Эквиваленты 1:", 0.1, 10.0, 1.0, key="e1") if use_equiv_bool else 1.0

        with col2:
            st.markdown("**Реагент 2**")
            if "Органическое + Органическое" in reaction_type:
                sm2 = st.text_input("SMILES реагента 2:", key="r2_sm", placeholder="CCO, CC(=O)O")
                r2_f = f"smiles:{sm2}" if sm2 else ""
                if sm2 and st.button("🔍 Показать структуру 2", key="show_r2"):
                    display_molecule(sm2, "Реагент 2")
            elif "Органическое + Неорганическое" in reaction_type:
                r2_f = st.text_input("Формула реагента 2 (неорг):", key="r2_f", placeholder="H2SO4, NaCl")
            else:
                r2_f = st.text_input("Формула реагента 2:", key="r2_f", placeholder="NaCl, H2SO4")
            coeff2 = st.number_input("Коэффициент 2:", 0.1, 10.0, 1.0, key="c2")
            mass2 = st.number_input("Масса 2 (г):", 0.0, 1000.0, 0.0, key="m2")
            equiv2 = st.number_input("Эквиваленты 2:", 0.1, 10.0, 1.0, key="e2") if use_equiv_bool else 1.0

        st.divider()
        st.subheader("📦 Продукты")
        num_prod = st.number_input("Количество продуктов:", 1, 5, 1, key="num_prod")
        products = []
        for i in range(num_prod):
            with st.expander(f"Продукт {i+1}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    p_type = st.selectbox(f"Тип продукта {i+1}:", ["Органический", "Неорганический"], key=f"ptype_{i}")
                    if p_type == "Органический":
                        p_sm = st.text_input(f"SMILES продукта {i+1}:", key=f"psm_{i}", placeholder="CC(=O)OCC, CCO")
                        p_f = f"smiles:{p_sm}" if p_sm else ""
                        if st.button(f"🔍 Показать структуру продукта {i+1}", key=f"btn_prod_{i}"):
                            if p_sm:
                                st.session_state.show_product_structure[f"prod_{i}"] = True
                            else:
                                st.warning(f"Введите SMILES для продукта {i+1}")
                    else:
                        p_f = st.text_input(f"Формула продукта {i+1}:", key=f"pf_{i}", placeholder="H2O, NaCl")
                        p_sm = None
                with col2:
                    p_coeff = st.number_input(f"Коэффициент {i+1}:", 0.1, 10.0, 1.0, key=f"pcoeff_{i}", step=0.1)
                products.append({'formula': p_f, 'coeff': p_coeff, 'name': f"Продукт {i+1}", 'type': p_type, 'smiles': p_sm})
                # Отображаем структуру, если флаг установлен
                if st.session_state.show_product_structure.get(f"prod_{i}", False) and p_sm:
                    display_molecule(p_sm, f"Структура продукта {i+1}")
                    st.session_state.show_product_structure[f"prod_{i}"] = False

        if st.button("🧮 Рассчитать реакцию", type="primary", use_container_width=True):
            if not r1_f or not r2_f:
                st.error("Введите оба реагента")
            else:
                calc = ReactionCalculator()
                calc.add_reagent("Реагент 1", r1_f, coeff1, mass1, equiv1)
                calc.add_reagent("Реагент 2", r2_f, coeff2, mass2, equiv2)
                for p in products:
                    if p['formula']:
                        calc.add_product(p['name'], p['formula'], p['coeff'])
                if not calc.products:
                    st.warning("Добавьте хотя бы один продукт")
                else:
                    res = calc.calculate(use_equiv_bool)
                    prod_res = calc.calculate_products(res, use_equiv_bool)
                    st.session_state.saved_reaction = calc.get_reaction_data()
                    st.subheader("📈 Результаты")
                    # Уравнение
                    reag_str = []
                    for r in calc.reagents:
                        f = r['calc']['formula_display']
                        c = r['coefficient']
                        reag_str.append(f"{c}{f}" if c!=1 else f)
                    prod_str = []
                    for p in calc.products:
                        f = p['calc']['formula_display']
                        c = p['coefficient']
                        prod_str.append(f"{c}{f}" if c!=1 else f)
                    st.info(f"{' + '.join(reag_str)} → {' + '.join(prod_str)}")
                    st.divider()
                    if res:
                        df_reag = pd.DataFrame([{
                            "Реагент": r['name'],
                            "Формула": r['formula_display'],
                            "Коэф": r['coefficient'],
                            "Масса (г)": r['mass'],
                            "M (г/моль)": f"{r['molar_mass']:.4f}",
                            "n (моль)": f"{r['moles']:.4f}",
                            "Лимитирующий": "⭐" if r.get('limiting') else ""
                        } for r in res])
                        st.dataframe(df_reag, use_container_width=True, hide_index=True)
                    if prod_res:
                        df_prod = pd.DataFrame([{
                            "Продукт": p['name'],
                            "Формула": p['formula_display'],
                            "Коэф": p['coefficient'],
                            "M (г/моль)": f"{p['molar_mass']:.4f}",
                            "Теор. n (моль)": f"{p['moles']:.4f}",
                            "Теор. масса (г)": f"{p['mass']:.2f}"
                        } for p in prod_res])
                        st.dataframe(df_prod, use_container_width=True, hide_index=True)
                        limiting = next((r for r in res if r.get('limiting')), None)
                        if limiting:
                            st.success(f"⭐ Лимитирующий реагент: {limiting['name']} ({limiting['formula_display']})")
                    st.success("Данные сохранены для расчёта по массе")
                    # Экспорт
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                        if res: pd.DataFrame(res).to_excel(writer, sheet_name='Реагенты', index=False)
                        if prod_res: pd.DataFrame(prod_res).to_excel(writer, sheet_name='Продукты', index=False)
                    st.download_button("📥 Скачать Excel", data=out.getvalue(), file_name=f"reaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", use_container_width=True)

    # --- РАСЧЁТ ПО ИЗВЕСТНОЙ МАССЕ ---
    with tab3:
        st.markdown('<div class="reaction-calc">⚖️ Расчет по известной массе</div>', unsafe_allow_html=True)
        if st.session_state.saved_reaction:
            data = st.session_state.saved_reaction
            # Уравнение
            reag_str = []
            for r in data['reagents']:
                f = r['calc']['formula_display']
                c = r['coefficient']
                reag_str.append(f"{c}{f}" if c!=1 else f)
            prod_str = []
            for p in data['products']:
                f = p['calc']['formula_display']
                c = p['coefficient']
                prod_str.append(f"{c}{f}" if c!=1 else f)
            st.info(f"{' + '.join(reag_str)} → {' + '.join(prod_str)}")
            # Список веществ с корректной молярной массой
            all_subs = []
            for r in data['reagents']:
                if r['calc']['molar_mass'] is not None:
                    all_subs.append({'name': r['name'], 'formula': r['calc']['formula_display'], 'type': 'reagent',
                                     'mass': r['calc']['molar_mass'], 'coeff': r['coefficient']})
            for p in data['products']:
                if p['calc']['molar_mass'] is not None:
                    all_subs.append({'name': p['name'], 'formula': p['calc']['formula_display'], 'type': 'product',
                                     'mass': p['calc']['molar_mass'], 'coeff': p['coefficient']})
            if not all_subs:
                st.error("Нет веществ с корректно определённой молярной массой. Проверьте формулы реагентов и продуктов.")
            else:
                opts = [f"{s['name']} ({s['formula']})" for s in all_subs]
                idx = st.selectbox("Выберите известное вещество:", range(len(opts)), format_func=lambda x: opts[x])
                known = all_subs[idx]
                known_mass = st.number_input(f"Масса {known['name']} ({known['formula']}) (г):", 0.0, 10000.0, 10.0, step=0.1)
                if st.button("Рассчитать", type="primary", use_container_width=True):
                    if known['mass']:
                        known_moles = known_mass / known['mass']
                        base = known_moles / known['coeff']
                        reagents_res = []
                        for r in data['reagents']:
                            if r['calc']['molar_mass'] is None:
                                continue
                            moles = base * r['coefficient']
                            mass = moles * r['calc']['molar_mass']
                            reagents_res.append({
                                "Реагент": r['name'],
                                "Формула": r['calc']['formula_display'],
                                "Коэф": r['coefficient'],
                                "М (г/моль)": f"{r['calc']['molar_mass']:.4f}",
                                "n (моль)": f"{moles:.4f}",
                                "Масса (г)": f"{mass:.2f}",
                                "Примечание": "📌 Исходное" if r['name'] == known['name'] else ""
                            })
                        products_res = []
                        for p in data['products']:
                            if p['calc']['molar_mass'] is None:
                                continue
                            moles = base * p['coefficient']
                            mass = moles * p['calc']['molar_mass']
                            products_res.append({
                                "Продукт": p['name'],
                                "Формула": p['calc']['formula_display'],
                                "Коэф": p['coefficient'],
                                "М (г/моль)": f"{p['calc']['molar_mass']:.4f}",
                                "n (моль)": f"{moles:.4f}",
                                "Масса (г)": f"{mass:.2f}"
                            })
                        st.subheader("📊 Результаты расчета")
                        if reagents_res:
                            st.markdown("#### Реагенты")
                            st.dataframe(pd.DataFrame(reagents_res), use_container_width=True, hide_index=True)
                        if products_res:
                            st.markdown("#### Продукты")
                            st.dataframe(pd.DataFrame(products_res), use_container_width=True, hide_index=True)
                        st.info(f"📌 **Сводка:** {known['name']} ({known['formula']}) — {known_mass:.2f} г → {known_moles:.4f} моль")
                        out = io.BytesIO()
                        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                            if reagents_res: pd.DataFrame(reagents_res).to_excel(writer, sheet_name='Реагенты', index=False)
                            if products_res: pd.DataFrame(products_res).to_excel(writer, sheet_name='Продукты', index=False)
                        st.download_button("📥 Скачать Excel", data=out.getvalue(), file_name=f"stoich_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", use_container_width=True)
                    else:
                        st.error("Не удалось рассчитать молярную массу выбранного вещества")
        else:
            st.warning("⚠️ Сначала рассчитайте реакцию во вкладке 'Расчет реакций'")
