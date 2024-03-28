import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Install necessary packages
# !pip install transformers sentencepiece

st.title('English to Tamil Translation')

# Load pre-trained model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

# Input text area for English article
article_en = st.text_area('Enter the English article here', 'U.N encourages wearing masks')

# Translate button
if st.button('Translate'):
    # Tokenize input article
    model_inputs = tokenizer(article_en, return_tensors="pt")

    # Generate translation from English to Hindi
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["ta_IN"]
    )

    # Decode generated tokens
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # Display translation
    st.header('Translated TamilArticle')
    st.write(translation[0])
