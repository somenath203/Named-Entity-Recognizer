import streamlit as st 
import spacy
from spacy import displacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd


st.title('Named Entity Recognizer')

st.write('Named Entity Recognition (NER) is like a smart highlighter that scans through text and highlights important words, such as people’s names, places, companies, and dates.')

st.write('')

with st.form(key='form_named_entity_recognition'):

    input_from_user = st.text_area('enter your input')

    model_options = st.selectbox('choose a model', ('Choose a model', 'Spacy\'s en_core_web_sm model', 'dslim/bert-base-NER model'))

    submit_button = st.form_submit_button('Submit')


if submit_button:

    if input_from_user == '':

        st.error('empty form submitted')
    
    else:

        if model_options == 'Choose a model':

            st.error('Please choose a model for named entity recognition')
    
        else:

            st.subheader('Result Analysis')

            if model_options == 'Choose a model':

                st.error('Please choose a model for Named Entity Recognition')

            elif model_options == 'Spacy\'s en_core_web_sm model':

                st.write('Model Used for Named Entity Recognition:')
                st.success(model_options)

                spacy_model = spacy.load('en_core_web_sm')

                res = spacy_model(input_from_user)

                st.write(f'Analysis of the detected entities from the text ==>')
                st.markdown(f'**{input_from_user}**')
                
                entities = [{'Entity': entity.text, 'Label of the Entity': entity.label_, 'Description of the Label': spacy.explain(entity.label_)} for entity in res.ents]

                df = pd.DataFrame(entities)

                st.table(df)

                st.write('Entites marked in the input text:')
                st.markdown(displacy.render(res, style='ent'), unsafe_allow_html=True)

            elif model_options == 'dslim/bert-base-NER model':

                st.write('Model Used for Named Entity Recognition:')
                st.success(model_options)
                
                tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
                model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

                bert_ner_model = pipeline('ner', model=model, tokenizer=tokenizer)

                res = bert_ner_model(input_from_user)

                abbreviations = {
                    "O": "Outside of a named entity",
                    "B-MISC": "Beginning of a miscellaneous entity right after another miscellaneous entity",
                    "I-MISC": "Miscellaneous entity",
                    "B-PER": "Beginning of a person’s name right after another person’s name",
                    "I-PER": "Person’s name",
                    "B-ORG": "Beginning of an organization right after another organization",
                    "I-ORG": "Organization",
                    "B-LOC": "Beginning of a location right after another location",
                    "I-LOC": "Location"
                }

                st.write(f'Analysis of the detected entities from the text ==>')
                st.markdown(f'**{input_from_user}**')
                
                entities = [{'Entity': input_from_user[entity['start']:entity['end']], 'Label of the Entity': entity['entity'], 'Description of the Label': abbreviations.get(entity['entity'])} for entity in res]

                df = pd.DataFrame(entities)

                st.table(df)
    

