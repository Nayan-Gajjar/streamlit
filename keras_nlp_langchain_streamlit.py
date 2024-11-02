"""
This script demonstrate how to deploy and serve RAG based chatbot using Streamlit.

In this setup, we’ll deploy a local server that will host a Streamlit-based GUI
(Graphical User Interface). This GUI will allow users to interact directly with
the Language Model (LLM) through an intuitive, user-friendly interface.

The local system will work as a server that provides real-time responses to the
user's inputs in the GUI.


Package Versions:
-----------------
python==3.11.10
keras_nlp==0.14.4
tf-keras==2.17.0
langchain==0.3.4
faiss-cpu==1.9.0
sentence-transformers==3.2.0
langchain-community==0.3.3
streamlit==1.39.0


Reference:
----------
https://docs.streamlit.io/get-started/installation
https://docs.streamlit.io/develop/api-reference
https://docs.streamlit.io/develop/concepts/architecture/run-your-app


To run the app:
---------------
streamlit run your_script.py --server.address 0.0.0.0 --server.port 8080
"""

# Import required packages
# ======================================================================================================================
import keras_nlp
import streamlit as st
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from typing import Any, Iterator, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.outputs import GenerationChunk
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


# Define global vaiables
# ======================================================================================================================

# Define keras_nlp model to be used with Langchain
# .............................................................................
gpt2 = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_extra_large_en")


# Define required class and functions
# ======================================================================================================================
class GPT2LLM(LLM):
    """
    A class which overrides LLM class from Langchain.
    This LLM class is useful when we want to define custom llm. This custom
    llm can be private/local API or private/local model as well.
    """
    num_output:int = 128

    @property
    def _llm_type(self) -> str:
        """
        Get the type of language model used by this chat model.
        Used for logging purposes only
        """
        return "GPT2 Large Model"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string.
        """
        print(f"{'*'*25} Prompt is \n {prompt}")

        response = gpt2.generate(prompt, max_length=self.num_output)

        return response
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        response = ""
        for token in gpt2.generate(prompt, max_length=self.num_output):
            response += token
            yield response


# Define global context for LLM and Embedings models
# ======================================================================================================================
llm = GPT2LLM()  # Using GPT2LLM as the language model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Load index and generate the asnwer
# ======================================================================================================================
# load index
# The index has been generated using keras_nlp_langchain.py script
index = FAISS.load_local("/home/nayan/NLP/RAG/Scripts/index", embed_model, allow_dangerous_deserialization=True)

# Set up the query engine using LangChain's RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.as_retriever()
)

# REF URL: https://docs.streamlit.io/develop/api-reference/widgets/st.text_area
txt = st.text_area("Question", "Ask questions to LLM!!!")

# Submission button
if st.button("Submit"):
    
    # Retrieve the answer for the submitted question
    response = qa_chain.invoke(txt)
    
    # Display the response
    st.write("Answer:", response["result"])