import streamlit as st
import numpy as np
import pandas as pd
import io
import re
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import streamlit.components.v1 as components
import base64

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(
    page_title="Peptide Analyzer Pro", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# --- СКРЫТИЕ ЭЛЕМЕНТОВ ИНТЕРФЕЙСА ---
hide_style = """
    <style>
    #GithubIcon {visibility: hidden;}
    div[data-testid="stStatusWidget"] {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    @media (max-width: 768px) {
        .main > div {padding: 0.5rem !important;}
        h1 {font-size: 1.8rem !important;}
        .stButton button {min-height: 44px !important;}
        .js-plotly-plot {max-height: 400px !important;}
    }
    
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    
    [data-testid="chatMessageUser"] .stChatMessage {
        background-color: #007bff;
        color: white;
    }
    
    .stButton button {
        border-radius: 20px;
        transition: all 0.3s;
    }
    
    .dev-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    
    .main .block-container {
        padding-top: 1rem;
    }
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# --- АВАТАРКИ ---
def get_user_avatar():
    avatar_svg = '''
    <svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <circle cx="20" cy="20" r="18" fill="#007bff" stroke="#0056b3" stroke-width="2"/>
        <path d="M16 11 L24 11 L26 21 L14 21 L16 11" fill="white" stroke="#2C3E50" stroke-width="1.5"/>
        <rect x="18" y="7" width="4" height="5" rx="1.5" fill="white" stroke="#2C3E50" stroke-width="1.5"/>
        <path d="M16 21 L24 21 L22 30 L18 30 L16 21" fill="#4ECDC4" opacity="0.9"/>
        <circle cx="19" cy="24" r="1.8" fill="white" opacity="0.9"/>
        <circle cx="22" cy="26" r="1.2" fill="white" opacity="0.9"/>
        <circle cx="16" cy="15" r="1.2" fill="#2C3E50"/>
        <circle cx="24" cy="15" r="1.2" fill="#2C3E50"/>
    </svg>
    '''
    return f"data:image/svg+xml;base64,{base64.b64encode(avatar_svg.encode('utf-8')).decode('utf-8')}"

def get_bot_avatar():
    avatar_svg = '''
    <svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <circle cx="20" cy="20" r="18" fill="#28a745" stroke="#1e7e34" stroke-width="2"/>
        <path d="M16 11 L24 11 L26 21 L14 21 L16 11" fill="white" stroke="#2C3E50" stroke-width="1.5"/>
        <rect x="18" y="7" width="4" height="5" rx="1.5" fill="white" stroke="#2C3E50" stroke-width="1.5"/>
        <path d="M16 21 L24 21 L22 30 L18 30 L16 21" fill="#FF6B6B" opacity="0.9"/>
        <circle cx="19" cy="24" r="1.8" fill="white" opacity="0.9"/>
        <circle cx="22" cy="26" r="1.2" fill="white" opacity="0.9"/>
        <circle cx="16" cy="15" r="2.2" fill="none" stroke="#2C3E50" stroke-width="1.5"/>
        <circle cx="24" cy="15" r="2.2" fill="none" stroke="#2C3E50" stroke-width="1.5"/>
    </svg>
    '''
    return f"data:image/svg+xml;base64,{base64.b64encode(avatar_svg.encode('utf-8')).decode('utf-8')}"

# --- ФУНКЦИИ РАСЧЕТА ---
def calculate_dihedral(p1, p2, p3, p4):
    """Расчет торсионного угла между четырьмя точками в 3D пространстве."""
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
    """Парсинг .mol2 файла для извлечения структуры и координат всех конформеров."""
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
    """Генерация SVG изображения молекулы с подписями атомов и ручным управлением камерой."""
    if not mol: 
        return "<div>Ошибка: молекула не загружена</div>"
    
    try:
        display_mol = Chem.RemoveHs(mol)
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(display_mol)
        
        canvas_width, canvas_height = 1000, 500
        d2d = rdMolDraw2D.MolDraw2DSVG(canvas_width, canvas_height)
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
        
        # Применяем трансформации (смещение и зум) через главный контейнер <g>
        transform = f"transform='translate({pan_x}, {pan_y}) scale({zoom})'"
        svg = svg.replace('<g>', f'<g {transform}>', 1)
        svg = re.sub(r'width=[\'"].*?[\'"]', 'width="100%"', svg)
        svg = re.sub(r'height=[\'"].*?[\'"]', 'height="100%"', svg)
        
        return f"<div style='background:white; border-radius:12px; border:1px solid #ddd; overflow:auto;'>{svg}</div>"
    except Exception as e:
        return f"<div style='color:red; padding:20px;'>Ошибка отрисовки: {str(e)}</div>"

def calculate_angles_for_node(conformers, phi_atoms, psi_atoms):
    """Расчет углов для одного узла по всем конформерам"""
    phi_vals = []
    psi_vals = []
    
    for conf in conformers:
        try:
            phi = calculate_dihedral(
                conf[phi_atoms[0]], conf[phi_atoms[1]], 
                conf[phi_atoms[2]], conf[phi_atoms[3]]
            )
            psi = calculate_dihedral(
                conf[psi_atoms[0]], conf[psi_atoms[1]], 
                conf[psi_atoms[2]], conf[psi_atoms[3]]
            )
            phi_vals.append(phi)
            psi_vals.append(psi)
        except Exception as e:
            phi_vals.append(0.0)
            psi_vals.append(0.0)
    
    return phi_vals, psi_vals

# --- ОТВЕТЫ БОТА ---
def get_bot_response(user_message):
    msg_lower = user_message.lower()
    
    if any(word in msg_lower for word in ["умеешь", "можешь", "что делаешь"]):
        return """
🔬 **Что я умею делать?**

📁 **Загрузка:** .mol2 файлы
🔬 **Расчет:** углы Phi и Psi для нескольких узлов
📊 **Графики:** отдельные графики для каждого узла
💾 **Сохранение:** Excel, HTML, SVG

⚠️ Я в разработке, но анализ углов работает!
"""
    
    elif any(word in msg_lower for word in ["привет", "здравствуй"]):
        return "👋 Привет! Загрузите .mol2 файл для анализа углов Phi и Psi."
    
    else:
        return """
⚠️ Я в разработке и не готов к глубокому общению.

**Что можно сделать:**
• Загрузить .mol2 файл
• Выбрать количество узлов
• Спросить "Что ты умеешь?"
"""

# --- ИНИЦИАЛИЗАЦИЯ ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "🧪 Загрузите .mol2 файл для анализа углов Phi и Psi."}
    ]

if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False

# --- ОСНОВНОЙ ИНТЕРФЕЙС ---
st.title("🧪 Peptide Analyzer Pro")

# Две колонки
chat_col, main_col = st.columns([1, 2])

with chat_col:
    st.markdown("### 💬 Чат")
    chat_container = st.container(height=450)
    
    with chat_container:
        for msg in st.session_state.messages[-10:]:
            avatar = get_bot_avatar() if msg["role"] == "assistant" else get_user_avatar()
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
    
    if prompt := st.chat_input("..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = get_bot_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with main_col:
    if not st.session_state.file_loaded:
        st.markdown("""
        <div class="dev-status">
            ⚠️ Приложение в разработке
        </div>
        """, unsafe_allow_html=True)
    
    # Настройки визуализации
    with st.expander("⚙️ Настройки", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            b_len = st.slider("Длина связей", 20, 150, 60)
            f_size = st.slider("Размер шрифта", 6, 30, 14)
        with col2:
            zoom_val = st.slider("Масштаб", 0.1, 5.0, 1.0)
        with col3:
            off_x = st.slider("Смещение X", -800, 800, 0)
            off_y = st.slider("Смещение Y", -800, 800, 0)
    
    uploaded_file = st.file_uploader("📁 Загрузите .mol2", type=['mol2'])
    
    if uploaded_file and not st.session_state.file_loaded:
        st.session_state.file_loaded = True
        st.rerun()
    
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        mol, conformers, heavy_atoms = parse_mol2_and_get_mol(file_content)
        
        if mol and conformers and heavy_atoms:
            st.subheader("🔬 Карта атомов")
            
            # Отрисовка молекулы
            svg_code = render_static_svg(mol, heavy_atoms, b_len, f_size, off_x, off_y, zoom_val)
            components.html(svg_code, height=520)

            st.subheader("⚛️ Настройка анализа")
            
            # Показываем количество конформеров
            st.info(f"📊 Найдено конформеров: {len(conformers)}")
            
            # Выбор количества узлов
            max_nodes = min(5, len(heavy_atoms) // 4)
            if max_nodes < 1:
                max_nodes = 1
                
            num_nodes = st.number_input("Количество узлов для анализа", 1, max_nodes, 1)
            
            # Создаем вкладки для каждого узла
            node_tabs = st.tabs([f"Узел {i+1}" for i in range(num_nodes)])
            
            # Словари для хранения выбранных атомов
            phi_atoms_dict = {}
            psi_atoms_dict = {}
            
            for i, tab in enumerate(node_tabs):
                with tab:
                    st.markdown(f"**Узел {i+1}**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Phi**")
                        phi_atoms_dict[i] = st.multiselect(
                            f"4 атома для Phi (узел {i+1})", 
                            options=heavy_atoms, 
                            max_selections=4, 
                            key=f"phi_node_{i}", 
                            placeholder="C, N, CA, C..."
                        )
                        if len(phi_atoms_dict[i]) == 4:
                            st.success(f"✓ {' → '.join(phi_atoms_dict[i])}")
                    
                    with col2:
                        st.markdown("**Psi**")
                        psi_atoms_dict[i] = st.multiselect(
                            f"4 атома для Psi (узел {i+1})", 
                            options=heavy_atoms, 
                            max_selections=4, 
                            key=f"psi_node_{i}", 
                            placeholder="N, CA, C, N..."
                        )
                        if len(psi_atoms_dict[i]) == 4:
                            st.success(f"✓ {' → '.join(psi_atoms_dict[i])}")
            
            # Проверяем готовность
            all_ready = True
            for i in range(num_nodes):
                if len(phi_atoms_dict.get(i, [])) != 4 or len(psi_atoms_dict.get(i, [])) != 4:
                    all_ready = False
                    break
            
            if all_ready:
                st.divider()
                
                # Расчет углов для всех узлов
                all_results = {}
                phi_psi_pairs = []
                
                for node_idx in range(num_nodes):
                    phi_vals, psi_vals = calculate_angles_for_node(
                        conformers, phi_atoms_dict[node_idx], psi_atoms_dict[node_idx]
                    )
                    all_results[f"node_{node_idx}"] = {
                        "phi": phi_vals,
                        "psi": psi_vals,
                        "phi_atoms": phi_atoms_dict[node_idx],
                        "psi_atoms": psi_atoms_dict[node_idx]
                    }
                    
                    # Для отладки - выводим первые значения
                    if len(phi_vals) > 0 and len(psi_vals) > 0:
                        phi_psi_pairs.append({
                            "node": node_idx + 1,
                            "first_phi": phi_vals[0],
                            "first_psi": psi_vals[0]
                        })
                
                # Показываем отладочную информацию (можно убрать позже)
                with st.expander("Отладка - первые значения углов"):
                    for item in phi_psi_pairs:
                        st.write(f"Узел {item['node']}: Phi={item['first_phi']}, Psi={item['first_psi']}")
                
                st.subheader("📊 Результаты")
                
                # Создаем вкладки для результатов
                result_tabs = st.tabs([f"Узел {i+1}" for i in range(num_nodes)])
                
                # Для каждого узла показываем таблицу и график
                for i, tab in enumerate(result_tabs):
                    with tab:
                        node_key = f"node_{i}"
                        
                        # Создаем DataFrame для текущего узла
                        node_df = pd.DataFrame({
                            "Конформер": range(1, len(conformers) + 1),
                            "Phi": all_results[node_key]["phi"],
                            "Psi": all_results[node_key]["psi"]
                        })
                        
                        # Показываем выбранные атомы
                        st.caption(f"Атомы Phi: {' → '.join(all_results[node_key]['phi_atoms'])}")
                        st.caption(f"Атомы Psi: {' → '.join(all_results[node_key]['psi_atoms'])}")
                        
                        col1, col2 = st.columns([1, 1.5])
                        
                        with col1:
                            st.dataframe(
                                node_df.style.format({"Phi": "{:.1f}", "Psi": "{:.1f}"}),
                                use_container_width=True, 
                                hide_index=True, 
                                height=400
                            )
                        
                        with col2:
                            fig = px.scatter(
                                node_df, 
                                x="Phi", 
                                y="Psi", 
                                text="Конформер",
                                title=f"Узел {i+1}",
                                labels={"Phi": "Phi (°)", "Psi": "Psi (°)"},
                                range_x=[-185, 185], 
                                range_y=[-185, 185],
                                template="plotly_white"
                            )
                            fig.update_traces(
                                marker=dict(size=10, color='royalblue'),
                                textposition='top center'
                            )
                            fig.update_layout(height=400)
                            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
                            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                html_data = pio.to_html(fig)
                                st.download_button(
                                    f"🌐 HTML", 
                                    html_data, 
                                    file_name=f"node_{i+1}.html", 
                                    mime="text/html",
                                    use_container_width=True
                                )
                            with col_b:
                                svg_data = pio.to_image(fig, format="svg")
                                st.download_button(
                                    f"🖼️ SVG", 
                                    svg_data, 
                                    file_name=f"node_{i+1}.svg", 
                                    mime="image/svg+xml",
                                    use_container_width=True
                                )
                
                # Создаем Excel с несколькими листами
                st.divider()
                
                # Подготавливаем данные для Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Лист для каждого узла
                    for i in range(num_nodes):
                        node_key = f"node_{i}"
                        node_df = pd.DataFrame({
                            "Конформер": range(1, len(conformers) + 1),
                            "Phi": all_results[node_key]["phi"],
                            "Psi": all_results[node_key]["psi"]
                        })
                        node_df.to_excel(writer, sheet_name=f'Узел_{i+1}', index=False)
                        
                        # Форматирование листа узла
                        worksheet = writer.sheets[f'Узел_{i+1}']
                        
                        # Создаем форматы
                        header_format = writer.book.add_format({
                            'bold': True,
                            'bg_color': '#4F81BD',
                            'font_color': 'white',
                            'border': 1
                        })
                        
                        number_format = writer.book.add_format({'num_format': '0.0'})
                        
                        # Применяем формат к заголовкам
                        for col_num, value in enumerate(node_df.columns.values):
                            worksheet.write(0, col_num, value, header_format)
                        
                        # Применяем числовой формат к колонкам Phi и Psi
                        worksheet.set_column('B:C', 15, number_format)
                        
                        # Добавляем информацию о выбранных атомах
                        worksheet.write(len(node_df) + 2, 0, "Выбранные атомы:", header_format)
                        worksheet.write(len(node_df) + 3, 0, f"Phi: {' → '.join(all_results[node_key]['phi_atoms'])}")
                        worksheet.write(len(node_df) + 4, 0, f"Psi: {' → '.join(all_results[node_key]['psi_atoms'])}")
                    
                    # Сводный лист
                    summary_data = []
                    for conf in range(1, len(conformers) + 1):
                        row = [conf]
                        for i in range(num_nodes):
                            node_key = f"node_{i}"
                            row.append(all_results[node_key]["phi"][conf-1])
                            row.append(all_results[node_key]["psi"][conf-1])
                        summary_data.append(row)
                    
                    columns = ["Конформер"]
                    for i in range(num_nodes):
                        columns.extend([f"Phi_узел{i+1}", f"Psi_узел{i+1}"])
                    
                    summary_df = pd.DataFrame(summary_data, columns=columns)
                    summary_df.to_excel(writer, sheet_name='Сводка', index=False)
                    
                    # Форматирование сводного листа
                    worksheet = writer.sheets['Сводка']
                    
                    # Применяем формат к заголовкам сводной таблицы
                    for col_num, value in enumerate(columns):
                        worksheet.write(0, col_num, value, header_format)
                    
                    # Применяем числовой формат ко всем колонкам кроме первой
                    for col_num in range(1, len(columns)):
                        worksheet.set_column(col_num, col_num, 15, number_format)
                
                st.download_button(
                    "📥 Скачать Excel (все узлы)", 
                    data=output.getvalue(), 
                    file_name=f"peptide_analysis_{num_nodes}_nodes.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.error("Не удалось загрузить молекулу из файла. Проверьте формат файла.")
