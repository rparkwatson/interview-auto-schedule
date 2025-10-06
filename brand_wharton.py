# brand_wharton.py
import streamlit as st
import altair as alt

# Official palette (from Wharton Identity Kit)
WHARTON_BLUE  = "#011F5B"  # Pantone 288
WHARTON_RED   = "#990000"  # Pantone 201
# A few secondary tones from the kit
PACIFIC_BLUE  = "#026CB5"
BAY_BLUE      = "#06AAFC"
EVENING_RED   = "#532A85"
MORNING_YEL   = "#D7BC6A"
MARINE_GRAY   = "#EEEDEA"
COLLEGE_GRAY  = "#B2B6A7"
NIGHT_STREET  = "#2D2C41"

def apply_wharton_brand(
    title: str,
    icon: str = "🏠",
    wide: bool = True,
    show_sidebar_logo: bool = False,
    logo_path: str | None = "assets/wharton-logo.png",  # supply only if you have rights to use it
):
    # MUST be first Streamlit call on the page
    st.set_page_config(page_title=title, page_icon=icon, layout=("wide" if wide else "centered"))

    # CSS: variables + button/spacing polish
    st.markdown(f"""
    <style>
      :root {{
        --w-blue:  {WHARTON_BLUE};
        --w-red:   {WHARTON_RED};
        --w-bg:    #FFFFFF;
        --w-surf:  {MARINE_GRAY};
        --w-text:  #0F172A;
        --w-radius: 14px;
      }}
      .block-container {{
        padding-top: 2.0rem; padding-bottom: 2.4rem;
      }}
      /* Headings */
      h1, h2, h3, h4, h5, h6 {{ color: var(--w-blue); }}
      /* Primary (CTA) buttons in Wharton Red */
      .stButton > button[kind="primary"],
      .stButton > button[data-testid="baseButton-primary"] {{
        background: var(--w-red) !important; color: #fff !important;
        border: 0 !important; border-radius: var(--w-radius) !important;
        padding: .6rem 1rem !important; box-shadow: none !important;
      }}
      /* Secondary buttons: outline in blue */
      .stButton > button[kind="secondary"],
      .stButton > button[data-testid="baseButton-secondary"] {{
        color: var(--w-blue) !important; border: 1px solid var(--w-blue) !important;
        background: #fff !important; border-radius: var(--w-radius) !important;
      }}
      /* Expander / cards */
      [data-testid="stExpander"] > div {{
        background: var(--w-surf); border-radius: var(--w-radius);
      }}
      /* Tables slightly rounded */
      .stDataFrame {{ border-radius: var(--w-radius); overflow: hidden; }}

      /* Typography: follow Wharton fallbacks (Acumin/Minion Pro → Arial/Georgia) */
      html, body, [class*="st-"] {{
        font-family: "Acumin Pro", "Acumin", Arial, Helvetica, "Segoe UI", system-ui, sans-serif !important;
        color: var(--w-text);
      }}
      .stMarkdown, .stText, p, li {{
        font-family: Georgia, "Minion Pro", "Times New Roman", Times, serif;
      }}
      /* Links-as-buttons utility */
      .w-link-btn {{
        display:inline-block; text-decoration:none; font-weight:600;
        background: {MORNING_YEL}; color:#111; padding:.55rem 1rem;
        border-radius: var(--w-radius);
      }}
      /* Sidebar */
      [data-testid="stSidebar"] {{ background: var(--w-surf); }}
    </style>
    """, unsafe_allow_html=True)

    # Optional: logo in sidebar (only if you have permission/licenses to use the mark)
    if show_sidebar_logo and logo_path:
        try:
            st.sidebar.image(logo_path, use_container_width=True)
        except Exception:
            pass

    _enable_altair_theme()


def _enable_altair_theme():
    # Brand Altair theme: blue-dominant palette with red as accent
    def wharton_theme():
        return {
            "config": {
                "view": {"continuousWidth": 420, "continuousHeight": 300},
                "font": "Acumin Pro, Arial, Helvetica, sans-serif",
                "background": "#FFFFFF",
                "title": {"fontSize": 16, "fontWeight": 600, "color": "#0F172A"},
                "axis":  {"labelColor": "#0F172A", "titleColor": "#0F172A"},
                "legend":{"labelColor": "#0F172A", "titleColor": "#0F172A"},
                "range": {
                    "category": [
                        "#011F5B", "#990000", "#026CB5", "#06AAFC",
                        "#532A85", "#2D2C41", "#B2B6A7", "#D7BC6A"
                    ]
                },
                "line": {"strokeWidth": 3},
                "bar": {"cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
            }
        }
    alt.themes.register("wharton", wharton_theme)
    alt.themes.enable("wharton")
