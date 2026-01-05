import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="AI Powered Dead Stock Prediction",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLOR_MAP = {
    'High': '#dc2626',
    'Medium': '#f59e0b',
    'Low': '#10b981'
}

kategori_map = {"Clothing": 0, "Electronics": 1, "Furniture": 2, "Groceries": 3, "Toys": 4}
kategori_listesi = list(kategori_map.keys())

@st.cache_resource
def get_pretrained_model():
    csv_path = "retail_store_inventory.csv"
    try:
        df_train = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"⚠️ Hata: '{csv_path}' dosyası bulunamadı. Lütfen 70.000 satırlık eğitim setini proje klasörüne ekleyin.")
        st.stop()

    # 2. KATEGORİLERİ SAYISALA ÇEVİRME
    # Eğer verinde bu sütun yoksa haritaya (kategori_map) göre dönüştürür.
    if 'Category_Code' not in df_train.columns:
        df_train['Category_Code'] = df_train['Category'].map(kategori_map).fillna(0)
    
    # 3. LABELING (ETİKETLEME)
    # Eğer 70binlik verinde 'is_dead_stock' diye bir sütun yoksa,
    # senin mantığına göre (skorlama) bu sütunu oluşturuyoruz.
    if 'is_dead_stock' not in df_train.columns:
        # Vektörel işlem (For döngüsü 70bin satırda yavaş çalışır, bu yöntem hızlıdır)
        conditions = [
            (df_train['Units Sold'] < 5),
            (df_train['Price'] > df_train['Competitor Pricing'] * 1.2),
            ((df_train['Inventory Level'] > 80) & (df_train['Units Sold'] < 10))
        ]
        
        # Senin skorlama mantığının aynısı:
        scores = (conditions[0].astype(int) * 4) + \
                 (conditions[1].astype(int) * 3) + \
                 (conditions[2].astype(int) * 3)
        
        # Skor 4 ve üzeriyse 1 (Dead Stock), değilse 0
        initial_labels = (scores >= 4).astype(int)
        
        # %10 oranında rastgele etiketleri bozuyoruz (Noise Injection)
        # np.random.seed(42) # Sabit sonuç isterseniz bunu açın
        random_noise = np.random.rand(len(df_train)) < 0.10 # %10 gürültü
        
        # Gürültü denk gelen yerleri tersine çevir (1 ise 0, 0 ise 1 yap)
        # abs(1 - label) işlemi 1'i 0, 0'ı 1 yapar.
        final_labels = np.where(random_noise, 1 - initial_labels, initial_labels)
        
        df_train['is_dead_stock'] = final_labels
    
    # 4. MODEL EĞİTİMİ
    feature_cols = ['Inventory Level', 'Price', 'Competitor Pricing', 'Discount', 'Units Sold', 'Category_Code']
    
    # Eksik veri varsa temizle
    df_train = df_train.dropna(subset=feature_cols)
    
    X = df_train[feature_cols]
    y = df_train['is_dead_stock']
    
    # n_jobs=-1 ile tüm işlemcileri kullanarak hızlı eğitir
    rf_model = RandomForestClassifier(n_estimators=100, random_state=12, n_jobs=-1)
    rf_model.fit(X, y)
    
    return rf_model, feature_cols

rf_model, feature_cols = get_pretrained_model()

with st.sidebar:
    st.title("AI Powered Dead Stock Prediction")
    st.write("Analiz yapmak için lütfen veri setinizi yükleyin.")
    st.divider()
    
    uploaded_file = st.file_uploader("CSV Dosyasını Yükle", type=["csv"])
    
    st.divider()
    if uploaded_file is None:
        st.info("⚠️ Veri bekleniyor...")
    else:
        st.success("✅ Veri yüklendi!")

if uploaded_file is None:
    st.header("👋 Hoşgeldiniz!")
    st.markdown("""
    **AI Powered Dead Stock Prediction** sistemine hoş geldiniz.
    
    Şu anda sistemde gösterilecek veri bulunmuyor. Yapay zeka modelini kullanmak için lütfen sol taraftan bir **CSV dosyası** yükleyin.
    
    **CSV Dosyanızda şu sütunlar olmalıdır:**
    - `Product ID`
    - `Product Name`
    - `Category`
    - `Inventory Level`
    - `Price`
    - `Competitor Pricing`
    - `Discount`
    - `Units Sold`
    """)
    
    st.divider()
    
    example_csv = pd.DataFrame([
        {
            'Product ID': 'URN-001', 'Product Name': 'Örnek Tişört', 'Category': 'Clothing', 
            'Inventory Level': 100, 'Price': 250, 'Competitor Pricing': 240, 'Discount': 0.0, 'Units Sold': 2
        },
        {
            'Product ID': 'URN-002', 'Product Name': 'Örnek Kulaklık', 'Category': 'Electronics', 
            'Inventory Level': 20, 'Price': 1500, 'Competitor Pricing': 1600, 'Discount': 0.1, 'Units Sold': 45
        }
    ])
    csv_template = example_csv.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Örnek CSV Şablonunu İndir",
        data=csv_template,
        file_name="ornek_stok_sablonu.csv",
        mime="text/csv",
    )

else:
    try:
        df = pd.read_csv(uploaded_file)
        
        required_cols = ['Inventory Level', 'Price', 'Competitor Pricing', 'Discount', 'Units Sold', 'Category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Hata: Yüklediğiniz dosyada şu sütunlar eksik: {', '.join(missing_cols)}")
            st.stop()
            
        if 'Category_Code' not in df.columns:
            df['Category_Code'] = df['Category'].map(kategori_map).fillna(0)
            
        X_input = df[feature_cols]
        probs = rf_model.predict_proba(X_input)[:, 1]
        
        df['Risk Score'] = (probs * 100).astype(int)
        
        def get_risk_level(score):
            if score >= 70: return 'High'
            elif score >= 30: return 'Medium'
            else: return 'Low'
        
        df['Risk Level'] = df['Risk Score'].apply(get_risk_level)
        
        df['History'] = df.apply(lambda x: np.random.randint(5, 50, 6).tolist(), axis=1)
        df['Forecast'] = df.apply(lambda x: [x['Inventory Level']] * 3, axis=1)

        page = st.radio("Görünüm Seçiniz:", ["Dashboard", "Envanter Listesi", "Ürün Detayları"], horizontal=True)
        st.divider()

        if page == "Dashboard":
            c1, c2, c3, c4 = st.columns(4)
            high_risk = len(df[df['Risk Level'] == 'High'])
            med_risk = len(df[df['Risk Level'] == 'Medium'])
            total_val = (df['Inventory Level'] * df['Price']).sum()
            
            c1.metric("🔴 Yüksek Riskli", high_risk, delta="Acil")
            c2.metric("🟡 Orta Riskli", med_risk)
            c3.metric("🔵 Stok Değeri", f"₺{total_val:,.0f}")
            c4.metric("🟢 Toplam Ürün", len(df))
            
            st.divider()
            
            g1, g2 = st.columns([2, 1])
            
            with g1:
                st.subheader("Risk Dağılımı")
                risk_counts = df['Risk Level'].value_counts().reindex(['High', 'Medium', 'Low']).fillna(0)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x=risk_counts.index, y=risk_counts.values, hue=risk_counts.index, palette=COLOR_MAP, ax=ax, legend=False)
                ax.set_ylabel("Adet")
                
                for i, v in enumerate(risk_counts.values):
                    if v > 0: ax.text(i, v + 0.1, str(int(v)), ha='center', fontweight='bold')
                
                st.pyplot(fig)
                
            with g2:
                st.subheader("Kategori Analizi")
                cat_counts = df['Category'].value_counts()
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
                st.pyplot(fig2)

        elif page == "Envanter Listesi":
            st.subheader("📋 Tüm Envanter")
            
            risk_filter = st.multiselect("Risk Filtresi", ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])
            filtered_df = df[df['Risk Level'].isin(risk_filter)]
            
            st.dataframe(
                filtered_df.sort_values(by="Risk Score", ascending=False),
                column_config={
                    "Risk Score": st.column_config.ProgressColumn("Risk Skoru", min_value=0, max_value=100, format="%d"),
                    "Price": st.column_config.NumberColumn("Fiyat", format="₺%d")
                },
                width="stretch"
            )

        elif page == "Ürün Detayları":
            st.subheader("🔎 Ürün Analizi")
            
            product_list = df['Product Name'].tolist() if 'Product Name' in df.columns else df['Product ID'].tolist()
            selected = st.selectbox("Ürün Seçin", product_list)
            
            if 'Product Name' in df.columns:
                row = df[df['Product Name'] == selected].iloc[0]
            else:
                row = df[df['Product ID'] == selected].iloc[0]
            
            dc1, dc2 = st.columns([1, 2])
            
            with dc1:
                color = COLOR_MAP.get(row['Risk Level'], '#gray')
                st.markdown(f"""
                <div style='background-color:{color};padding:20px;border-radius:10px;color:white;text-align:center'>
                    <h1>{row['Risk Score']}</h1>
                    <h3>{row['Risk Level']} Risk</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.info(f"**Stok:** {row['Inventory Level']} Adet")
                st.info(f"**Fiyat:** ₺{row['Price']}")
                
            with dc2:
                st.write("**🤖 AI Önerisi:**")
                if row['Risk Level'] == 'High':
                    st.error("Bu ürün Dead Stock olma yolunda! Acil kampanya veya %30 indirim önerilir. Stok alımını durdurun.")
                elif row['Risk Level'] == 'Medium':
                    st.warning("Satış hızı yavaşlıyor. Rakip fiyatlarını kontrol edin ve ürün görünürlüğünü artırın.")
                else:
                    st.success("Performans harika. Stok seviyesi ideal. Yeni sipariş planlayabilirsiniz.")

    except Exception as e:
        st.error(f"Dosya okunurken bir hata oluştu: {e}")
