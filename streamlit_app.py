import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

# Sample DataFrame
df = pd.read_excel("final.xlsx")

def create_association_rules(dataframe):
    # Your existing code for association rules
    transactions_str = df.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Count')
    my_basket = transactions_str.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)

    def encode(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    my_basket_sets = my_basket.applymap(encode)
    frequent_items = apriori(my_basket_sets, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)
    rules.sort_values('confidence', ascending=False, inplace=True)

    antecedents = [', '.join(item) for item in rules['antecedents']]
    consequents = [', '.join(item) for item in rules['consequents']]

    # Remove duplicate antecedents
    antecedents = list(set(antecedents))
    return antecedents

# Main Streamlit app
def main():
    st.title("Association Rules Streamlit App")

    if st.button("Generate Association Rules"):
        antecedents = create_association_rules(df)
        st.success("Association rules generated!")

        # Create a dropdown menu to select antecedents
        selected_antecedent = st.selectbox("Select your 1st Liquor", antecedents)
    
        # Display consequents based on the selected antecedent
        consequents_str = f"Consequents for {selected_antecedent}: [consequent_1, consequent_2, ...]"
        st.text(consequents_str)

if __name__ == "__main__":
    main()

