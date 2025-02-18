import streamlit as st
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from utils import process_documents, get_retriever
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


def get_custom_prompt():
    """Define and return the custom prompt template."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Bạn là một trợ lý về quy trình, hãy đưa ra thông tin chính xác nhất cho người hỏihỏi. Thực hiện theo các nguyên tắc sau:\n"
"1. Trả lời câu hỏi chỉ bằng thông tin từ các tệp PDF đã tải lên.\n"
"2. Sử dụng ngôn ngữ tiếng việt, đơn giản, rõ ràng.\n"
"3. Nếu câu trả lời không có trong tài liệu, hãy nói: 'Tôi không thể tìm thấy thông tin liên quan trong các tài liệu được cung cấp.'\n"
"4. Không suy đoán, giả định hoặc bịa đặt thông tin.\n"
"5. Duy trì giọng điệu chuyên nghiệp và sắp xếp câu trả lời rõ ràng (ví dụ: gạch đầu dòng, giải thích từng bước).\n"
"6. Khuyến khích các câu hỏi tiếp theo bằng cách hỏi xem có cần làm rõ thêm không.\n"
"7. Cung cấp các ví dụ để làm rõ các khái niệm khi hữu ích.\n"
"8. Giữ câu trả lời ngắn gọn, tập trung và thân thiện với bài kiểm tra.\n"
"9. Đưa ra câu trả lời bằng tiếng việt, không được dùng ngôn ngữ khác."

        ),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Cung cấp một câu trả lời chính xác và có cấu trúc tốt dựa trên ngữ cảnh trên. Đảm bảo câu trả lời của bạn dễ hiểu, bao gồm các ví dụ khi cần thiết và được định dạng theo cách mà người hỏi có thể sử dụng câu trả lời đó. Nếu có, hãy hỏi xem người hỏi có cần làm rõ thêm không."
        )
    ])

def initialize_qa_chain():
    if not st.session_state.qa_chain and st.session_state.vector_store:
        llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0.3)
        retriever = get_retriever()
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": get_custom_prompt()}
        )
    return st.session_state.qa_chain

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None



def display_sidebar():
    with st.sidebar:
        pdfs = st.file_uploader(
            "Tải File PDF", 
            type="pdf",
            accept_multiple_files=True 
        )
        
        if st.button("Xử lý văn bản"):
            if not pdfs:
                st.warning("Vui lòng tải file PDF")
                return

            try:
                with st.spinner("Đang xử lý văn bản... Điều này có thể mất vài phút."):
                    vector_store = process_documents(pdfs)
                    st.session_state.vector_store = vector_store
                    st.session_state.qa_chain = None  
                st.success("Văn bản đã xử lý")  
            except Exception as e:
                st.error(f"Lỗi xử lý văn bản: {str(e)}")  


def chat_interface():
    st.title("Chatbot")
    st.markdown("Hỏi đáp quy trình nội bộ")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Bạn muốn hỏi gì?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            
            with st.spinner("Đang tìm kiếm thông tin"):
                try:
                    qa_chain = initialize_qa_chain()
                    
                    if not qa_chain:
                        full_response = "Vui lòng tạo văn bản bằng việc tải File PDF."
                    else:
                        response = qa_chain.invoke({"query": prompt})
                        full_response = response["result"]
                except Exception as e:
                    full_response = f"Error: {str(e)}"

            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def main():
    initialize_session_state()
    display_sidebar()
    chat_interface()

if __name__ == "__main__":
    main()