import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. AYARLAR VE BAŞLIK
# =============================================================================
st.set_page_config(page_title="Dead Stock AI", page_icon="📦", layout="wide")

st.title("📦 Dead Stock Tahmin Sistemi")
st.markdown("""
Bu sistem, yapay zeka kullanarak ürünlerin **Ölü Stok (Dead Stock)** olma riskini hesaplar.
Veri girişi yaparak veya CSV yükleyerek analiz yapabilirsiniz.
""")

# =============================================================================
# 2. KATEGORİ AYARLARI (SENİN LİSTEN)
# =============================================================================
# Senin verdiğin kategorileri sayısal kodlara eşliyoruz
kategori_map = {
    "Clothing": 0,
    "Electronics": 1,
    "Furniture": 2,
    "Groceries": 3,
    "Toys": 4
}
kategori_listesi = list(kategori_map.keys())

# =============================================================================
# 3. MODELİ EĞİT (GÜNCELLENMİŞ KATEGORİLERLE)
# =============================================================================
@st.cache_resource
def modeli_egit():
    # --- 1. Sentetik Veri Oluşturma ---
    np.random.seed(42)
    n_samples = 500
    
    final_df = pd.DataFrame({
        'Product ID': [f'PROD_{i}' for i in range(n_samples)],
        'Inventory Level': np.random.randint(0, 100, n_samples),
        'Price': np.random.randint(10, 2000, n_samples),
        'Discount': np.random.choice([0, 0.1, 0.2], n_samples),
        'Units Sold': np.random.randint(0, 50, n_samples),
        
        # SADECE SENİN KATEGORİLERİNİ KULLANIYORUZ
        'Category': np.random.choice(kategori_listesi, n_samples),
        
        # Etiketleme için kullanılan yardımcı sütunlar (Eğitime girmez)
        'is_dead_stock': np.random.choice([0, 1], n_samples)
    })

    # Kategorileri sayıya çevirelim (Mapping kullanarak)
    final_df['Category_Code'] = final_df['Category'].map(kategori_map)

    # --- 2. EĞİTİM MANTIĞI ---
    # Modelin öğreneceği sütunlar
    feature_cols = ['Inventory Level', 'Price', 'Discount', 'Units Sold', 'Category_Code']
    
    X = final_df[feature_cols]
    y = final_df['is_dead_stock']
    
    # Train/Test Split (Senin parametrelerinle: rs=56, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=56, stratify=y
    )
    
    # Model Kurulumu (Senin parametrelerinle: rs=12)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=12)
    rf_model.fit(X_train, y_train)
    
    return rf_model, feature_cols

# Modeli hafızaya yükle
model, feature_cols = modeli_egit()

# =============================================================================
# 4. YAN MENÜ VE GİRİŞLER
# =============================================================================
st.sidebar.header("Veri Giriş Yöntemi")
giris_yontemi = st.sidebar.radio("Seçiniz:", ["Tek Ürün (Manuel)", "Toplu Analiz (CSV Yükle)"])

# --- SENARYO A: MANUEL GİRİŞ ---
if giris_yontemi == "Tek Ürün (Manuel)":
    st.sidebar.divider()
    st.sidebar.subheader("Ürün Özellikleri")
    
    # SADELEŞTİRİLMİŞ GİRDİLER
    inventory = st.sidebar.slider("Stok Seviyesi (Inventory Level)", 0, 500, 50)
    price = st.sidebar.number_input("Fiyat (Price)", 1, 10000, 100)
    units_sold = st.sidebar.number_input("Satış Adedi (Units Sold)", 0, 1000, 20)
    discount = st.sidebar.selectbox("İndirim (Discount)", [0.0, 0.1, 0.2, 0.3, 0.5])
    
    # Güncellenmiş Kategori Listesi
    cat_name = st.sidebar.selectbox("Kategori", kategori_listesi)
    
    if st.sidebar.button("Risk Analizi Yap", type="primary"):
        # Veriyi hazırla
        input_data = pd.DataFrame([{
            'Inventory Level': inventory,
            'Price': price,
            'Discount': discount,
            'Units Sold': units_sold,
            'Category_Code': kategori_map[cat_name] # Seçilen ismin kodunu bulur (Örn: Toys -> 4)
        }])
        
        # Sütun sırasını garantiye al
        input_data = input_data[feature_cols]
        
        # Tahmin
        prob = model.predict_proba(input_data)[0][1]
        
        st.divider()
        col1, col2 = st.columns(2)
        
        col1.metric("Risk Skoru", f"%{prob*100:.2f}")
        
        if prob > 0.5:
            col2.error("Durum: 🔴 DEAD STOCK RİSKİ")
            st.warning("⚠️ **Öneri:** Bu ürün kategorisinde (%s) stok eritme kampanyası yapın." % cat_name)
        else:
            col2.success("Durum: 🟢 GÜVENLİ")
            st.info("✅ **Öneri:** Stok seviyesi ideal.")

# --- SENARYO B: CSV YÜKLEME ---
else:
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader("CSV Dosyasını Yükle", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ Dosya Yüklendi! {len(df_upload)} satır analiz ediliyor...")
            
            # --- Ön İşleme ---
            # Eğer dosyada 'Category' sütunu varsa ve içindeki değerler (Clothing, Toys vb.) ise
            # Bunları otomatik olarak sayıya çeviriyoruz.
            if 'Category' in df_upload.columns and 'Category_Code' not in df_upload.columns:
                # Bilinmeyen kategori gelirse hata vermesin diye .fillna(0) ekledik
                df_upload['Category_Code'] = df_upload['Category'].map(kategori_map).fillna(0)
            
            # Eksik sütun kontrolü
            missing = [col for col in feature_cols if col not in df_upload.columns]
            
            if missing:
                st.error(f"❌ Hata: CSV dosyanızda şu sütunlar eksik: {missing}")
                st.info("Gerekli sütunlar: Inventory Level, Price, Discount, Units Sold, Category")
            else:
                # Tahmin
                X_new = df_upload[feature_cols]
                probs = model.predict_proba(X_new)[:, 1]
                
                # Sonuçları ekle
                df_upload['Dead_Stock_Risk_%'] = (probs * 100).round(2)
                df_upload['Tahmin'] = df_upload['Dead_Stock_Risk_%'].apply(lambda x: 'RİSKLİ' if x > 50 else 'GÜVENLİ')
                
                # Raporlama
                st.dataframe(df_upload.sort_values(by='Dead_Stock_Risk_%', ascending=False))
                
                # İndirme Butonu
                csv = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Sonuçları İndir", csv, "sonuclar.csv", "text/csv")
                
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
