import pandas as pd 

def handle_negative_values(df, cols_to_check):
    print("\n🔍 Vérification des valeurs négatives :")
    for col in cols_to_check:
        count_neg = (df[col] < 0).sum()
        if count_neg > 0:
            print(f"⚠️ {count_neg} valeur(s) négative(s) détectée(s) dans '{col}' — remplacement par NaN.")
            df.loc[df[col] < 0, col] = pd.NA
        else:
            print(f"✅ Pas de valeurs négatives dans '{col}'")
    return df