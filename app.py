import os
import streamlit as st
import pickle
import pdfplumber
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate

with st.sidebar:
    st.title('LLM Chat Bot')
    st.markdown('''
    ## About
    This is an LLM-powered chatbot built using :
    - [Streamlit]
    - [LangChain]
    - [OpenAI] LLM Model
    
    ''')
    st.write('AAA')

load_dotenv()


def main():
    st.header("Hello")

    pdf_files = st.file_uploader("Upload Your File", type='pdf', accept_multiple_files=True)
    st.button('Process')

    if pdf_files:

        responses = []
        pdf_names = []
        excel_file_name = "responses.xlsx"

        questionList = [
            "What is the invoice number?",
            "What is the date?",
            "What are the total price?",
            "What is the Company's GSTIN/UIN ?",
            "Who is the Consignee company ?",
            "What is the Consignee company's address ?",
            "What are the goods ?",
            "What are the Unit price for each goods. "
        ]

        for pdf in pdf_files:
            pdf_reader = PdfReader(pdf)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            st.write(len(text))
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=100,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)
            st.write(chunks)

            # embedding

            store_name = pdf.name[:-4]

            if os.path.exists(f"{store_name}.pk1"):
                with open(f"{store_name}.pk1", "rb") as f:
                    VectorStone = pickle.load(f)
                # st.write('Embeddings loaded from the disk')
            else:
                embeddings = OpenAIEmbeddings()
                VectorStone = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pk1", "wb") as f:
                    pickle.dump(VectorStone, f)
                # st.write('Embeddings Computation Completed')

            # accept query
            # query = st.text_input("Ask questions about your PDF:")
            prompt = """ 
            The content provided is extracted from a invoice pdf. 
            Please answer the following questions and obey the following rule.
            
            RULE:
            1.If cannot answer question based on the information provided , say 'i don't know'. 
            2.Add '@@@@' in each response's end. 
            3.Answer questions in short and concise.
            4.Don't repeat my questions in your response.
            
            Questions:
            
            """

            for question in questionList:
                prompt += f"{question}\n"

            if prompt:
                docs = VectorStone.similarity_search(query=prompt)
                llm = ChatOpenAI(model="gpt-3.5-turbo",
                                 temperature=0,
                                 model_name='text-davinci-003'
                                 )

                chain = load_qa_chain(llm=llm, chain_type="stuff")

                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=prompt)
                    print(cb)

                st.write('-------response--------')
                st.write(response)
                st.write(len(prompt))
                st.write(len(response))

                response_list = response.split('@@@@')
                response_list.pop()
                st.write('-------response_list--------')
                st.write(response_list)
                # while len(response_list) < len(questionList):
                #     response_list.append("N/A")

                # split 'is'
                a = []
                for i in range(len(response_list) - 2):
                    a.append(response_list[i].split(" is ")[-1].strip("."))

                # split 'are'
                b = []
                for i in range(len(response_list) - 2, len(response_list)):
                    r = response_list[i].split(" are ")[-1].strip(".")
                    r = r.replace(", and ", ", ")
                    r = r.replace("and ", ", ")
                    b.append(r)

                st.write('-------b--------')
                st.write(b)

                responses.append([pdf.name] + a + b)
                pdf_names.append(pdf.name)

                st.write('-------responses--------')
                st.write(responses)


            if os.path.exists(excel_file_name):
                os.remove(excel_file_name)

            column_names = ['PDF Name'] + questionList

            if os.path.exists(excel_file_name):
                with pd.ExcelWriter(excel_file_name, mode='a', engine='openpyxl') as writer:
                    df = pd.DataFrame(responses, columns=column_names)
                    df.to_excel(writer, index=False, header=False)
            else:
                response_df = pd.DataFrame(responses, columns=column_names)
                response_df.to_excel(excel_file_name, index=False)

            st.write("Responses saved to Excel file:", excel_file_name)


if __name__ == '__main__':
    main()
