import requests, datetime, time, re
import streamlit as st
import fitz
import io
import tiktoken


#
# VARIABLES

if 'max_words' not in st.session_state:
    st.session_state.max_words = 512

if 'entity_range' not in st.session_state:
    st.session_state.entity_range = 5

if 'content_category' not in st.session_state:
    st.session_state.content_category = ""

if 'iterations' not in st.session_state:
    st.session_state.iterations = 5

COMPLETION_URL = f"{st.secrets['CODEGPT_API_URL']}chat/completions"

MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB

enc = tiktoken.encoding_for_model("gpt-4")

PROMPT = """\
You will generate increasingly concise, entity-dense summaries of the above Article.

Repeat the following 2 steps 5 times.

Step 1. Identify 1-5 informative Entities (";" delimited) from the Article which are missing from the previous summary. Step 2. Write a new, denser summary of identified critical length which covers every entity and details from the previous summary plus the Missing Entities.

A Missing Entity is:
- Relevant: to the main story.
- Specific: descriptive yet concise (5 words or fewer).
- Novel: not in the previous summary.
- Factual: present in the Article.
- Anywhere: located anywhere in the Article.

Guidelines:
- The first summary should be long yet highly non-specific, containing little information beyond the identified entities marked as missing. Use overly verbose language and filler (e.g., 'This article discusses') to reach ~512 words.
- Make every word count: re-write the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of unnecessary phrases like 'the article discusses'.
- The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
- Missing Entities can appear anywhere in the new summary.
- Never drop existing Entities from the previous summary. If space cannot be made, add fewer new Entities.

Remember, use the exact same number of words for each summary.

Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are " Missing_Entities" and " Denser_Summary".

# Article: \n\n"""

PROMPT_COD = """\
As an expert copy-writer, you will write increasingly concise, entity-dense summaries of the above Article. The initial summary should be under {max_words} words and contain {entity_range} informative Descriptive Entities from the {content_category}.

A Descriptive Entity is:
- Relevant: to the main story.
- Specific: descriptive yet concise (5 words or fewer).
- Faithful: present in the {content_category}.
- Anywhere: located anywhere in the {content_category}.

# Your Summarization Process
- Read through the {content_category} and the all the below sections to get an understanding of the task.
- Pick {entity_range} informative Descriptive Entities from the {content_category} (";" delimited, do not add spaces).
- In your output JSON list of dictionaries, write an initial summary of max {max_words} words containing the Entities.
- You now have `[{{"missing_entities": "...", "denser_summary": "..."}}]`

Then, repeat the below 2 steps {iterations} times:
- Step 1. In a new dict in the same list, identify {entity_range} new informative Descriptive Entities from the {content_category} which are missing from the previously generated summary.
- Step 2. Write a new, denser summary of identical length which covers every Entity and detail from the previous summary plus the new Missing Entities.

A Missing Entity is:
- An informative Descriptive Entity from the {content_category} as defined above.
- Novel: not in the previous summary.

# Guidelines
- The first summary should be long (max {max_words} words) yet highly non-specific, containing little information beyond the Entities marked as missing. Use overly verbose language and fillers (e.g., "this {content_category} discusses") to reach ~{max_words} words.
- Make every word count: re-write the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the {content_category} discusses".
- The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the {content_category}.
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
- You're finished when your JSON list has 1+{iterations} dictionaries of increasing density.

# IMPORTANT
- Remember, to keep each summary to max {max_words} words.
- Never remove Entities or details. Only add more from the {content_category}.
- Do not discuss the {content_category} itself, focus on the content: informative Descriptive Entities, and details.
- Remember, if you're overusing filler phrases in later summaries, or discussing the {content_category} itself, not its contents, choose more informative Descriptive Entities and include more details from the {content_category}.
- Answer with a minified JSON list of dictionaries with keys "missing_entities" and "denser_summary".

## Example output
[{{"missing_entities": "ent1;ent2", "denser_summary": "<vague initial summary with entities 'ent1','ent2'>"}}, {{"missing_entities": "ent3", "denser_summary": "denser summary with 'ent1','ent2','ent3'"}}, ...]

# Articles\n\n
"""


#
# MAIN

st.set_page_config(
    page_title="CodeGPT – Dense Summarization",
    page_icon=":open_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('CodeGPT – Dense Summarization')
st.write('This app extracts text from uploaded PDF files and generates dense summaries using the CodeGPT API.\n\n---')

def tokenize(text: str):
    text_tokens = enc.encode(text)
    tokens = len(text_tokens)
    return tokens

# Function to call the API with the given agent ID and message
def get_summary(
        file_name: str = "",
        file_content: str = ""
    ):

    start_time = datetime.datetime.now()
    end_time = None

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {st.secrets['CODEGPT_APIKEY']}"  # Replace with your actual API key
        # "CodeGPT-Org-Id": "",  # Replace with your actual Org ID
    }

    messages = [
        {
            "role": "user",
            "content": f"<ARTICLE>{PROMPT_COD.format( max_words=st.session_state.max_words, entity_range=st.session_state.entity_range, content_category=st.session_state.content_category, iterations=st.session_state.iterations)}\n{'# ' +file_name if file_name else ''}\n{file_content}</ARTICLE>"
        }
    ]

    payload = {
        "agentId": st.secrets['CODEGPT_AGENTID'],
        "messages": messages,
        "format": "text",
        "stream": False
    }

    # Make the POST request and return the JSON response
    try:
        get_response = requests.post(COMPLETION_URL, json=payload, headers=headers)
        print(get_response)

        if get_response.status_code == 200:

            response = get_response.json()

            end_time = datetime.datetime.now()

            print(f"Time taken: {(end_time - start_time).total_seconds()}s")

            return {
                "tokens": tokenize(response),
                "name": file_name,
                "content": f"{response}"
            }

    except Exception as e:
        print(f"API Error: {e}")
        return {"error": f"API Error: {e}"}

def stream_data(text: str):

    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

    return word

def get_denser_summary(text: str):

    denser_summary = re.findall(r'"denser_summary": "(.*?)"', text)
    return_string = ''

    for i in denser_summary:
        return_string += i + '\n'

    return return_string

#
# APP LAYOUT
st.sidebar.write('**Summarization Settings**')
st.session_state.content_category = st.sidebar.text_input('Main Subject', key='content_category_input' , value=st.session_state.content_category or 'Main Subject' )
st.session_state.max_words = st.sidebar.select_slider('Max Words', key='max_words_input', options=[128, 256, 512, 1024])

st.sidebar.write('---')

st.sidebar.write('**Upload Files**')

uploaded_files = st.sidebar.file_uploader(
    "Files",
    accept_multiple_files=True,
    type='pdf'
)


documents = []

if uploaded_files:
    # Iterate over each uploaded file
    for uploaded_file in uploaded_files:
        try:
            # Read the content of the file as bytes
            pdf_bytes = uploaded_file.read()
            # Open the PDF file using the PyMuPDF library
            pdf_document = fitz.open(stream=io.BytesIO(pdf_bytes))
            # Initialize an empty string to accumulate text from all pages
            pages = ""
            # Loop through each page in the PDF document
            for page in range(len(pdf_document)):
                page = pdf_document[page]
                # Extract text from the current page
                text = page.get_text()
                # Append the text to the pages string
                pages += text
            # Create a dictionary with token count, file name, and content
            tokens = tokenize(f"{PROMPT} # {uploaded_file.name}\n\n{pages}")
            document = {
                "name": uploaded_file.name,
                "tokens": tokens,
                "times": (tokens // 8000) + 1,
                "pages": len(pdf_document),
                "token_per_page": tokens // len(pdf_document),
                "content": pages
            }
            # Add the document data to the documents list
            documents.append(document)
        except Exception as e:
            # If an error occurs, display it in the Streamlit app
            st.error(f"PDF File Error: {e}")
    # Display the extracted data using the data editor widget
    with st.status(f"Extracted text from {len(documents)} PDF Files") as status:
        st.data_editor(documents, use_container_width=True, key='documents')
        status.update(
            label=f"1. Extracting text – Click to Expand | Total Tokens: {sum([doc['tokens'] for doc in documents])}",
            state="complete",
            expanded=False
            )


summaries = []
final_summary = []
summaries_string = ""

# What is New Order
if len(documents) > 0:
    st.sidebar.write('---')
    if st.sidebar.button(
        'Start Summarization',
        key='summarize',
        use_container_width=True, type='primary'
    ):

        if len(summaries) == 0:

            with st.status(f"Summarizing {len(documents)} Documents") as status:

                for document in documents:

                    status.update(
                        label=f"Summarizing **{document['name']}** – {document['tokens']} Tokens",
                        state="running",
                        expanded=False
                    )

                    create_summary = get_summary(
                        document['name'],
                        document['content']
                    )

                    summaries.append(create_summary)

                    print(create_summary)

                st.data_editor(summaries, use_container_width=True)

                status.update(
                    label=f"Summarize each document – Click to Expand",
                    state="complete",
                    expanded=False
                )



        if len(final_summary) == 0:

            with st.status(f"Merging {len(summaries)} Summaries") as status:

                for summary in summaries:
                    summaries_string += f"## {summary['name']}\n{summary['content']}\n\n"

                status.update(
                    label=f"**Final Summary** – {tokenize(summaries_string)} Tokens",
                    state="running",
                    expanded=False
                )

                create_final_summary = get_summary(
                    "",
                    summaries_string
                )

                final_summary.append(create_final_summary)

                st.data_editor(final_summary, use_container_width=True)

                print(create_final_summary)

                status.update(
                    label=f"3. Final Summary – Click to Expand",
                    state="complete",
                    expanded=False
                )

            st.write('---')
            st.write(f"**Final Summary**")

            print(f'{final_summary} Tokens')

            st.write_stream(stream_data(get_denser_summary(final_summary[0]['content'])))
