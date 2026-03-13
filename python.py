import streamlit as st
import numpy as np
import pandas as pd
import io
import re
import plotly.express as px
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import streamlit.components.v1 as components

# --- 1. ФУНКЦИИ РАСЧЕТА ---
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
    # Возвращаем float, округленный до 1 знака
    return round(float(np.degrees(np.arctan2(y, x))), 1)

def parse_mol2_and_get_mol(file_content):
    """Парсинг .mol2 файла для извлечения структуры и координат всех конформеров."""
    mol = Chem.MolFromMol2Block(file_content, removeHs=False)
    conformers, current_coords, heavy_atom_names = [], {}, []
    lines = file_content.splitlines()
    is_atom = False
    for line in lines:
        if line.startswith('@<TRIPOS>MOLECULE'):
            if current_coords: conformers.append(current_coords)
            current_coords, is_atom = {}, False
        elif line.startswith('@<TRIPOS>ATOM'): is_atom = True
        elif line.startswith(('@<TRIPOS>BOND', '@<TRIPOS>SUB')): is_atom = False
        elif is_atom:
            p = line.split()
            if len(p) >= 6:
                name = p[1]
                current_coords[name] = np.array([float(p[2]), float(p[3]), float(p[4])])
                if name not in heavy_atom_names and not name.upper().startswith('H'):
                    heavy_atom_names.append(name)
    if current_coords: conformers.append(current_coords)
    return mol, conformers, heavy_atom_names

def render_static_svg(mol, heavy_atom_names, bond_len, font_size, pan_x, pan_y, zoom):
    """Генерация SVG изображения молекулы с подписями атомов и ручным управлением камерой."""
    if not mol: return None
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
    
    return f"<div style='background:white; border-radius:12px; border:1px solid #ddd;'>{svg}</div>"

# --- 2. ИНТЕРФЕЙС ---
st.set_page_config(page_title="Peptide Analyzer Pro", layout="wide")
st.title("🧬 Анализ углов Phi (φ) и Psi (ψ)")

# Боковая панель настроек
st.sidebar.header("🎨 Визуализация")
b_len = st.sidebar.slider("Длина связей", 20, 150, 60)
f_size = st.sidebar.slider("Размер шрифта", 6, 30, 14)
st.sidebar.markdown("---")
st.sidebar.header("🕹️ Управление камерой")
zoom_val = st.sidebar.slider("Масштаб (Zoom)", 0.1, 5.0, 1.0)
off_x = st.sidebar.slider("Смещение по X", -800, 800, 0)
off_y = st.sidebar.slider("Смещение по Y", -800, 800, 0)

uploaded_file = st.file_uploader("Загрузите файл .mol2", type=['mol2'])

if uploaded_file:
    file_content = uploaded_file.read().decode("utf-8")
    mol, conformers, heavy_atoms = parse_mol2_and_get_mol(file_content)
    
    if mol:
        # Блок 1: Интерактивная карта
        st.subheader("1. Карта атомов")
        svg_code = render_static_svg(mol, heavy_atoms, b_len, f_size, off_x, off_y, zoom_val)
        components.html(svg_code, height=520)

        # Блок 2: Выбор атомов вручную
        st.subheader("2. Выбор атомов для расчета")
        sel_col1, sel_col2 = st.columns(2)
        
        with sel_col1:
            st.info("Укажите 4 атома для **Phi (φ)**")
            phi_at = st.multiselect("Атомы Phi:", options=heavy_atoms, max_selections=4, key="phi_s")
        
        with sel_col2:
            st.info("Укажите 4 атома для **Psi (ψ)**")
            psi_at = st.multiselect("Атомы Psi:", options=heavy_atoms, max_selections=4, key="psi_s")

        # Блок 3: Расчеты, Таблица и График
        if len(phi_at) == 4 and len(psi_at) == 4:
            st.divider()
            
            # Проводим расчеты по всем конформерам
            phi_vals = [calculate_dihedral(c[phi_at[0]], c[phi_at[1]], c[phi_at[2]], c[phi_at[3]]) for c in conformers]
            psi_vals = [calculate_dihedral(c[psi_at[0]], c[psi_at[1]], c[psi_at[2]], c[psi_at[3]]) for c in conformers]
            
            # Создаем DataFrame
            res_df = pd.DataFrame({
                "Конформер": range(1, len(phi_vals) + 1),
                "Phi (φ)": phi_vals,
                "Psi (ψ)": psi_vals
            })

            data_col, plot_col = st.columns([1, 2])
            
            with data_col:
                st.subheader("📄 Результаты")
                
                # Принудительное отображение .0 для всех чисел в таблице Streamlit
                st.dataframe(
                    res_df.style.format({"Phi (φ)": "{:.1f}", "Psi (ψ)": "{:.1f}"}),
                    use_container_width=True, 
                    hide_index=True, 
                    height=500
                )
                
                # Генерация Excel с сохранением формата
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    res_df.to_excel(writer, index=False, sheet_name='Angles')
                    workbook  = writer.book
                    worksheet = writer.sheets['Angles']
                    # Формат ячеек для Excel (всегда показывать один знак после запятой)
                    fmt = workbook.add_format({'num_format': '0.0'})
                    worksheet.set_column('B:C', 15, fmt)
                
                st.download_button(
                    "📥 Скачать отчет (.xlsx)", 
                    data=output.getvalue(), 
                    file_name="phi_psi_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with plot_col:
                st.subheader("📈 Двумерный график")
                
                # Создаем векторный интерактивный график
                fig = px.scatter(
                    res_df, 
                    x="Phi (φ)", 
                    y="Psi (ψ)", 
                    text="Конформер",
                    labels={"Phi (φ)": "Угол Phi (°)", "Psi (ψ)": "Угол Psi (°)"},
                    hover_data={"Конформер": True, "Phi (φ)": ":.1f", "Psi (ψ)": ":.1f"},
                    template="plotly_white"
                )
                
                fig.update_traces(
                    marker=dict(size=12, color='royalblue', line=dict(width=1, color='white')),
                    textposition='top center'
                )
                
                # Настройка осей и зума
                fig.update_layout(
                    xaxis=dict(range=[-185, 185], dtick=45, gridcolor='#f0f0f0'),
                    yaxis=dict(range=[-185, 185], dtick=45, gridcolor='#f0f0f0'),
                    height=550,
                    dragmode='pan' # Режим перемещения "рукой" по умолчанию
                )
                
                # Вспомогательные линии осей
                fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.2)
                fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.2)
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 **Управление:** Используйте колесико мыши для зума. Кнопки в углу графика позволяют сбросить вид.")
        else:
            st.warning("⚠️ Выберите по 4 атома для Phi и Psi, чтобы сформировать расчеты.")