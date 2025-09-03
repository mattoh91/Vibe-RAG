import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional

# Configuration
BACKEND_URL = "http://localhost:12000"
API_BASE = f"{BACKEND_URL}/api/v1"

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .status-success {
        color: #27ae60;
        font-weight: bold;
    }
    .status-error {
        color: #e74c3c;
        font-weight: bold;
    }
    .status-warning {
        color: #f39c12;
        font-weight: bold;
    }
    .document-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .source-card {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f0f8ff;
    }
</style>
""", unsafe_allow_html=True)

def check_backend_health() -> bool:
    """Check if backend is healthy"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_llm_providers() -> List[Dict]:
    """Get available LLM providers"""
    try:
        response = requests.get(f"{API_BASE}/llm/providers")
        if response.status_code == 200:
            return response.json()["providers"]
        return []
    except:
        return []

def configure_llm(provider: str, api_key: str, endpoint: str = None, model: str = None, api_version: str = None, deployment_name: str = None) -> Dict:
    """Configure LLM provider"""
    try:
        payload = {
            "provider": provider,
            "api_key": api_key,
            "model_name": model
        }
        if endpoint:
            payload["endpoint"] = endpoint
        if api_version:
            payload["api_version"] = api_version
        if deployment_name:
            payload["deployment_name"] = deployment_name
            
        response = requests.post(f"{API_BASE}/llm/configure", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def test_llm_connection() -> Dict:
    """Test LLM connection"""
    try:
        response = requests.post(f"{API_BASE}/llm/test")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def upload_document(file) -> Dict:
    """Upload a document"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE}/documents/upload", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_documents() -> List[Dict]:
    """Get list of documents"""
    try:
        response = requests.get(f"{API_BASE}/documents/")
        if response.status_code == 200:
            return response.json()["documents"]
        return []
    except:
        return []

def delete_document(document_id: str) -> bool:
    """Delete a document"""
    try:
        response = requests.delete(f"{API_BASE}/documents/{document_id}")
        return response.status_code == 200
    except:
        return False

def query_documents(query: str, max_results: int = 5, use_reranking: bool = True, document_ids: List[str] = None) -> Dict:
    """Query documents"""
    try:
        payload = {
            "query": query,
            "max_results": max_results,
            "use_reranking": use_reranking
        }
        if document_ids:
            payload["document_ids"] = document_ids
            
        response = requests.post(f"{API_BASE}/query/", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    # Header
    st.markdown('<div class="main-header">📚 RAG Document Assistant</div>', unsafe_allow_html=True)
    
    # Check backend health
    if not check_backend_health():
        st.error("❌ Backend service is not available. Please ensure the FastAPI server is running on port 12000.")
        st.stop()
    
    st.success("✅ Backend service is healthy")
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        
        # LLM Configuration
        st.subheader("LLM Provider Setup")
        
        providers = get_llm_providers()
        if providers:
            provider_options = {p["name"]: p["id"] for p in providers}
            selected_provider_name = st.selectbox("Select LLM Provider", list(provider_options.keys()))
            selected_provider = provider_options[selected_provider_name]
            
            # Find provider details
            provider_info = next(p for p in providers if p["id"] == selected_provider)
            
            # API Key input
            api_key = st.text_input("API Key", type="password", help="Enter your API key for the selected provider")
            
            # Azure OpenAI specific fields
            endpoint = None
            api_version = None
            deployment_name = None
            
            if selected_provider == "azure_openai":
                endpoint = st.text_input("Azure Endpoint", 
                                        placeholder="https://your-resource.openai.azure.com/",
                                        help="Enter your Azure OpenAI endpoint URL")
                api_version = st.text_input("API Version", 
                                          value="2024-10-01-preview",
                                          help="Azure OpenAI API version")
                deployment_name = st.text_input("Deployment Name", 
                                               placeholder="your-deployment-name",
                                               help="Your Azure OpenAI deployment name (e.g., masdkp-openai-gpt-4o)")
                selected_model = st.selectbox("Model", ["gpt-4o", "gpt-4", "gpt-35-turbo", "gpt-4-turbo"])
            else:
                # Other providers
                if provider_info.get("requires_endpoint"):
                    endpoint = st.text_input("Endpoint URL", help="Enter the endpoint URL")
                
                # Model selection
                model_options = provider_info.get("default_models", [])
                if model_options:
                    selected_model = st.selectbox("Model", model_options)
                else:
                    selected_model = st.text_input("Model Name", help="Enter the model name")
            
            # Configure button
            if st.button("Configure LLM"):
                if api_key:
                    # Validate Azure OpenAI specific fields
                    if selected_provider == "azure_openai":
                        if not endpoint or not deployment_name:
                            st.error("Please fill in all Azure OpenAI fields (Endpoint and Deployment Name)")
                            st.stop()
                    
                    with st.spinner("Configuring LLM..."):
                        result = configure_llm(selected_provider, api_key, endpoint, selected_model, api_version, deployment_name)
                        if "error" in result:
                            st.error(f"Configuration failed: {result['error']}")
                        else:
                            st.success("LLM configured successfully!")
                            st.session_state.llm_configured = True
                else:
                    st.error("Please enter an API key")
            
            # Test connection button
            if st.button("Test Connection"):
                with st.spinner("Testing connection..."):
                    result = test_llm_connection()
                    if "error" in result:
                        st.error(f"Connection test failed: {result['error']}")
                    elif result.get("success"):
                        st.success(f"✅ Connected to {result['provider']} ({result.get('model', 'Unknown model')})")
                    else:
                        st.error(f"❌ Connection failed: {result.get('message', 'Unknown error')}")
        
        st.divider()
        
        # Query Settings
        st.subheader("Query Settings")
        max_results = st.slider("Max Results", 1, 10, 5)
        use_reranking = st.checkbox("Use Reranking", value=True, help="Use cross-encoder reranking for better results")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="section-header">📄 Document Management</div>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader("Upload PDF Document", type="pdf", help="Upload a PDF document for RAG")
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    result = upload_document(uploaded_file)
                    if "error" in result:
                        st.error(f"Upload failed: {result['error']}")
                    else:
                        st.success(f"Document processed successfully! Document ID: {result['document_id']}")
                        st.rerun()
        
        # Document list
        st.subheader("Uploaded Documents")
        documents = get_documents()
        
        if documents:
            for doc in documents:
                with st.container():
                    st.markdown(f"""
                    <div class="document-card">
                        <strong>{doc['filename']}</strong><br>
                        <small>Status: <span class="status-{'success' if doc['status'] == 'completed' else 'warning'}">{doc['status']}</span></small><br>
                        <small>Pages: {doc.get('page_count', 'N/A')} | Chunks: {doc.get('chunk_count', 'N/A')}</small><br>
                        <small>Uploaded: {doc['upload_time'][:19]}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Delete", key=f"delete_{doc['document_id']}"):
                        if delete_document(doc['document_id']):
                            st.success("Document deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to delete document")
        else:
            st.info("No documents uploaded yet")
    
    with col2:
        st.markdown('<div class="section-header">💬 Chat with Documents</div>', unsafe_allow_html=True)
        
        # Query interface
        if not documents:
            st.info("Please upload some documents first to start querying")
        else:
            # Document selection
            doc_options = {"All Documents": None}
            doc_options.update({doc['filename']: doc['document_id'] for doc in documents})
            
            selected_docs = st.multiselect(
                "Select Documents (optional)", 
                list(doc_options.keys()),
                help="Leave empty to search all documents"
            )
            
            # Convert selection to document IDs
            selected_doc_ids = None
            if selected_docs and "All Documents" not in selected_docs:
                selected_doc_ids = [doc_options[doc] for doc in selected_docs if doc != "All Documents"]
            
            # Query input
            query = st.text_area("Ask a question about your documents:", height=100)
            
            if st.button("Ask Question", type="primary"):
                if query.strip():
                    with st.spinner("Searching documents and generating response..."):
                        result = query_documents(
                            query, 
                            max_results=max_results, 
                            use_reranking=use_reranking,
                            document_ids=selected_doc_ids
                        )
                        
                        if "error" in result:
                            st.error(f"Query failed: {result['error']}")
                        else:
                            # Display response
                            st.subheader("Response")
                            st.write(result["response"])
                            
                            # Display sources
                            if result.get("sources"):
                                st.subheader("Sources")
                                for i, source in enumerate(result["sources"], 1):
                                    with st.expander(f"Source {i}: {source['document_name']} (Page {source.get('page_number', 'N/A')})"):
                                        st.markdown(f"""
                                        <div class="source-card">
                                            <strong>Relevance Score:</strong> {source['relevance_score']:.3f}<br>
                                            <strong>Text:</strong><br>
                                            {source['chunk_text']}
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Display processing time
                            st.caption(f"Processing time: {result['processing_time']:.2f} seconds")
                else:
                    st.warning("Please enter a question")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <small>RAG Document Assistant - Built with FastAPI, Streamlit, and advanced NLP models</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()