import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from sqlalchemy import create_engine
from utils import Data_extractor,RAG,TimeTableProcessor
import bcrypt

#hasher = stauth.Hasher("99230040570")
st.set_page_config(page_title="Faculty Meeting Scheduler", layout="wide")
# Authentication configuration
with st.empty():
    st.image(r"landsacpe kare logo.jpeg",width=900)
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

def sign_in(details=config['credentials']["usernames"]):
    with st.form("Sign up"):
        st.header("Sign up")
        st.warning("Use only KLU mail id")
        email = st.text_input("Enter your email:")
        name=st.text_input("Enter your username:")
        password=st.text_input("Password:",type="password")
        submit=st.form_submit_button("Sign up")
        if submit :
            if not email.endswith("@klu.ac.in"):
                st.error("Only @klu.ac.in emails are allowed!")
            elif email in details:
                st.warning("This email is already registered.")
            else:
                # Hash Password and Store
                hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                config["credentials"]["usernames"][email] = {
                    "email": email,
                    "name": name,
                    "password": hashed_password
                }
                # Save to YAML
                with open("config.yaml", "w") as file:
                    yaml.dump(config, file)
                st.success("Registration Successful! You can now log in.")
def login():
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key']
        )
    # Authentication flow
    authenticator.login(location="main",single_session=True,fields={"Username":"Email"})  # No unpacking

    if st.session_state["authentication_status"]:
        st.sidebar.empty()
        st.sidebar.title(f"Welcome {st.session_state['name']}")
        if st.session_state['username'] != 'bharathinukurthi1@gmail.com':
            pages = {"Chat Interface": chat_page}
        else:
            pages = {"Chat Interface": chat_page,"Upload Timetable":upload_page}
        st.sidebar.title("Navigation")
        
        choice = st.sidebar.radio("Go to", list(pages.keys()))
        authenticator.logout(location="sidebar")
        pages[choice]()
        
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your credentials")
    

USERNAME = "root"   # Replace with your MySQL username
PASSWORD = "99230040570"   # Replace with your MySQL password
HOST = "localhost"           # Change if the database is hosted remotely
PORT = "3306"                # Default MySQL port
DATABASE = "faculty_meet"   # Replace with your database name

# Create SQLAlchemy Engine
engine = create_engine(f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")

# Course mapping and processing classes
inverse_course_mapping={'Statistics for Engineers': 'Statis',
 'Database Management Systems IC -': 'DBMS',
 'Java Programming IC -': 'Java',
 'Computer Architecture and Organization': 'CAO',
 'Excel Skills': 'EXSEL',
 'Machine Learning IC -': 'ML',
 'Digital Principles and System Design IC -': 'DPSD',
 'University Elective': 'UE',
"Pattern and Anomaly Detection": "PAD",
    "Pattern and Anomaly Detection Lab": "PAD Lab",
    "Computer Networks": "CN",
    "Automata and Compiler Design": "ACD",
    "Foundation on Innovation and Entrepreneurship": "FIE",
    "Design Project II": "EXSEL",
    "Secured Computing": "SC"
 }  # From notebook
 # Full class implementation

# Streamlit app structure
def main():
    with st.sidebar.empty():
        page1={"Login":login,"Sign up":sign_in}
        choice1=st.radio("Procees by:",list(page1.keys()))
    page1[choice1]()
    
def upload_page():
    
    st.header("PDF Timetable Upload")
    uploaded_file = st.file_uploader("Upload timetable PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing timetable..."):
            # Save and process PDF
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            processor = TimeTableProcessor(
                Data_extractor("temp.pdf",inverse_course_mapping).extracted,inverse_course_mapping
            )
            dbs = processor.process_all()
            # Update database
            for name, df in dbs.items():
                df.to_sql(name, con=engine, if_exists='replace', index=False)
            st.success("Timetable processed successfully!")

def chat_page():
    st.header("Faculty Meeting Query")
    query = st.text_input("Ask about faculty availability:")
    if query:
        with st.spinner("Analyzing schedule..."):
            # Generate and execute query
            response = RAG(query)
            st.write(response)

if __name__ == "__main__":
    main()