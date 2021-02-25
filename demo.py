from demo_utils import *

def predict(text):
    text = rdrsegmenter.tokenize(text)
    text = ' '.join([' '.join(x) for x in text])

    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    outputs = model.generate(input_ids, 
                             attention_mask=attention_mask,
                             max_length = configs['decoder_max_length'],
                             early_stopping= configs['early_stopping'],
                             num_beams= configs['num_beams'], 
                             no_repeat_ngram_size= configs['no_repeat_ngram_size'])

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_str[0]

st.title("Vietnamese text summarization")

st.subheader("Enter the text you'd like to summarize.")
text = st.text_input('Enter text')

pred = predict(text)
st.header("Results")
st.subheader("Summary")
st.write(pred)