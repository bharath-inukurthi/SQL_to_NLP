import os

from langchain.prompts import (ChatPromptTemplate,
                                SystemMessagePromptTemplate,
                                HumanMessagePromptTemplate,
                                FewShotChatMessagePromptTemplate
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.sql.vector_sql import get_result_from_sqldb
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re
import pandas as pd
import numpy as np
import pdfplumber as reader



class Data_extractor:
    def compatibility(self,page)->bool:
        if len(page.extract_tables())>2:
            return True
        return False
    def __init__(self,path:str,inverse_course_mapping):
        self.path=path
        self.extracted=[]
        self.mapping=inverse_course_mapping
        self.process()
    def process(self)->dict:
        with reader.open(r"{}".format(self.path)) as pdf:
            i=0
            for page in pdf.pages:
                if self.compatibility(page):
                    
                    #class_details=self.get_coordinator(page)
                    courses_details=self.get_course_details(page)
                    schedule=self.get_schedule(page,courses_details)
                    i+=1
                    print(i)
                    self.extracted.append(#{"class_details":class_details,
                            {"course_details":courses_details,
                            "schedule":schedule})
        return self.extracted
    def extract_course_room(self,cell):
        if isinstance(cell, str):  # Ensure it's a string
            parts = cell.split("\n")  # Split by newline
            course_name = parts[0].strip()  # First part is course name
            room_no = parts[1].strip() if len(parts) > 1 else None  # Second part is room number

            # Apply regex to ensure only valid course names
            match = re.match(r"([A-Za-z\s&]+)", course_name)
            if match:
                course_name = match.group(1).strip()

            return course_name, room_no
        return None, None  # If cell is not string, return None
    def convert_to_24hr(self,time_slot):
        if time_slot=="12.00-":
            time_slot="12.00-1.00"
        time_slot = time_slot.replace("\n", "").strip()  # Remove newlines and spaces
        start, end = time_slot.split("-")  # Split into start and end times
        start, end = start.strip(), end.strip()  # Trim spaces

        # Convert start time
        start_hour = int(start.split(".")[0])  # Extract hour part
        meridian = "PM" if start_hour<8 or start_hour==12 else "AM"
        start_24 = pd.to_datetime(f"{start} {meridian}", format="%I.%M %p").strftime("%H:%M")

        # Convert end time
        end_hour = int(end.split(".")[0])
        meridian = "PM" if end_hour<8 or end_hour==12 else "AM"
        end_24 = pd.to_datetime(f"{end} {meridian}", format="%I.%M %p").strftime("%H:%M")

        return f"{start_24}-{end_24}"

# Apply function to the 'Time Slot' column

    def get_schedule(self,page,courses)->dict:
        # Convert to Pandas DataFrame
        tables=page.extract_tables()
        df = pd.DataFrame(tables[1]).replace(["", "None"], np.nan).dropna(how="all")

        # Extract the time slots from the first row
        time_slots = df.iloc[0, 1:].tolist()  # First row, excluding the first column

        # Define a function to clean and split course & room number
        

        # Iterate over each row in the DataFrame
        processed_data = []
        for index, row in df.iterrows():
            if index != 0:  # Skip the first row (time slots row)
                day = row[0]  # First column is the day
                for i in range(1, len(row)):
                    course, room = self.extract_course_room(row[i])  # Extract course and room number
                    if course:  # Only add valid entries
                        processed_data.append([day, time_slots[i-1], course, room])  # Assign correct time slot
        final_df = pd.DataFrame(processed_data, columns=["Day", "Time Slot", "Course Name", "Room No"])
        course_details=pd.DataFrame(courses)
        course_details["Course Name"]=course_details["Course Name"].map(self.mapping)
        final_df=final_df.merge(course_details,on="Course Name",how="inner")
        final_df.drop(columns=["Course Name"],inplace=True)
        final_df["Time Slot"]=final_df["Time Slot"].apply(self.convert_to_24hr)
        return final_df
    
    def get_coordinator(self,page)->dict:
        text=page.extract_text()
        pattern = r"SLOT:\s*(SLOT\s*\d+).*?SECTION\s*–\s*(S\d+).*?(?:Class Coordinator|Mr\.|Ms\.)\s*([A-Za-z.\s-]+)"
        # Find matches
        match = re.search(pattern, text, re.DOTALL)
        if match:
 
            slot = match.group(1) # SLOT value
            
            section = match.group(2) # SECTION value
            
            coordinator = match.group(3) # Class Coordinator name
            incharge_details={
        "slot": slot,
        "section":section,
        "coordinator": coordinator,
    }
        return incharge_details
    
    def get_course_details(self,page)->dict:
        text=page.extract_text()
        courses=[]
        text = re.sub(r"(\b(Dr\.|Mr\.|Ms\.|Mrs\.)\S*)\n(\d+\.)?", r"\1 ", text)  # Fix split names
        text = re.sub(r"\n(?=[A-Z]+\s+[A-Z]+)", " ", text)  # Merge faculty names

        # Updated Regex Pattern (fixes multiple faculty merge issue)
        pattern = r"(\d{3}[A-Z]{3}\d{4})\s+([\w\s&\-]+?)\s+[A-Z-]+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(Dr\.|Mr\.|Ms\.|Mrs\.)?\s*([A-Z][A-Za-z\s.]+)?(?:\s*\n\d+\.)?"

        # Find all matches
        matches = re.findall(pattern, text)

        for course_code, course_name, title, faculty_name in matches:
            # Clean up faculty name (remove unwanted numbers and extra spaces)
            faculty = f"{title} {faculty_name}".strip() if title else faculty_name.strip()
            faculty = re.sub(r"\n\d+\.\s*", " ", faculty)  # Remove numbers inside names
            details={"course code":course_code,
                         "Course Name":course_name.strip(),
                                    "Faculty":faculty if faculty else 'N/A'
                        }
            courses.append(details)
        return courses



class TimeTableProcessor:
    def __init__(self, extracted_data, inverse_course_mapping):
        self.extracted_data = extracted_data
        self.inverse_course_mapping = inverse_course_mapping
    
    def create_section_db(self):
        section_db = pd.DataFrame([item["class_details"] for item in self.extracted_data])
        section_db = section_db[["section", "slot"]].reset_index().rename(columns={"index": "Section_id"})
        return section_db
    
    def create_subject_db(self):
        subject_table = [pd.DataFrame(item["course_details"])[["course code", "Course Name"]] for item in self.extracted_data]
        subject_db = pd.concat(subject_table).rename(columns={"Course Name": "Course"}).drop_duplicates().reset_index(drop=True)
        subject_db["Course Name"] = subject_db["Course"].map(self.inverse_course_mapping)
        return subject_db
    
    def create_faculty_db(self):
        faculty_table = [pd.DataFrame(item["course_details"])["Faculty"] for item in self.extracted_data]
        faculty_db = pd.DataFrame(pd.Series([j for i in faculty_table for j in i]).unique(), columns=["Faculty"])
        faculty_db = faculty_db.reset_index().rename(columns={"index": "Faculty_id"})
        return faculty_db
    
    def create_faculty_subject_db(self, faculty_db):
        faculty_subject_table = [pd.DataFrame(item["course_details"])[["course code", "Faculty"]] for item in self.extracted_data]
        df = pd.concat(faculty_subject_table, axis=0)
        faculty_subject_data = df.merge(faculty_db, on="Faculty")
        faculty_subject_data = faculty_subject_data.drop_duplicates(subset=["course code", "Faculty"], keep="first")
        faculty_subject_data = faculty_subject_data.reset_index(drop=True).reset_index().rename(columns={"index": "fs_id"})
        return faculty_subject_data
    
    def create_days_db(self):
        day_table = [pd.DataFrame(item["schedule"])["Day"] for item in self.extracted_data]
        days_db = pd.DataFrame(pd.concat(day_table, axis=0).unique(), columns=["Day"]).reset_index().rename(columns={"index": "day_id"})
        return days_db
    
    def create_slots_db(self):
        slots_table = [pd.DataFrame(item["schedule"])["Time Slot"] for item in self.extracted_data]
        slots_db = pd.DataFrame(pd.concat(slots_table, axis=0).unique(), columns=["Time Slot"]).reset_index().rename(columns={"index": "Time_slot_id"})[:-1]
        return slots_db
    
    def create_room_db(self):
        room_table = [pd.DataFrame(item["schedule"])["Room No"] for item in self.extracted_data]
        room_db = pd.DataFrame(pd.Series([j for i in room_table for j in i]).replace({"Comp": "Computer block"}).unique(), columns=["Room No"])
        room_db = room_db.reset_index().rename(columns={"index": "Room ID"})
        return room_db
    
    def create_time_table_db(self, faculty_subject_data, room_db, slots_db, days_db):
        time_table_data = []
        for item in self.extracted_data:
            df = pd.DataFrame(item["schedule"]).merge(faculty_subject_data, on=["course code", 'Faculty'], how="inner").drop(columns=["course code", 'Faculty', "Faculty_id"])
            df = df.merge(room_db, on="Room No", how="inner").drop(columns=["Room No"])
            df = df.merge(slots_db, on="Time Slot", how="inner").drop(columns=["Time Slot"])
            df = df.merge(days_db, on="Day", how="inner").drop(columns=["Day"])
            time_table_data.append(df)
        time_table_db = pd.concat(time_table_data).reset_index(drop=True).reset_index().rename(columns={"index": "Time_table_id"})
        return time_table_db
    
    def process_all(self):
        #section_db = self.create_section_db()
        subject_db = self.create_subject_db()
        faculty_db = self.create_faculty_db()
        faculty_subject_db = self.create_faculty_subject_db(faculty_db)
        days_db = self.create_days_db()
        slots_db = self.create_slots_db()
        room_db = self.create_room_db()
        time_table_db = self.create_time_table_db(faculty_subject_db, room_db, slots_db, days_db)
        
        return {
            #"section_db": section_db,
            "subject_db": subject_db,
            "faculty_db": faculty_db,
            "faculty_subject_db": faculty_subject_db,
            "days_db": days_db,
            "slots_db": slots_db,
            "room_db": room_db,
            "time_table_db": time_table_db
        }


def RAG(input_query):
    sys_template = SystemMessagePromptTemplate.from_template("""You are a helpful assistant tasked with generating plain SQL queries based on user questions.
    Guidelines:
    1. Generate a syntactically correct MYSQL query that answers the user's question.
    2. Provide only the SQL query as the output.
    - Do not include any markdown formatting (e.g., ```sql or ```)
    Do not add prefixes like "SQLQuery:" or any other additional text.
    4. Order results by a relevant column, if applicable, to return the most interesting examples.
    5. Never select all columns; include only those relevant to the question.
    6. Use only the column names from the schema provided. Avoid querying non-existent columns.
    7. **If the user's question cannot be answered using the provided schema or data, respond with: "We don't have related data in the database".*

    Only use the following tables:
    {table_info}

    Notice:
    - dont limit the output
    - The response contains only the plain SQL query without prefixes or suffixes.
    - No markdown or additional formatting.
    - write a sql query as plain character string and avoid giving answer in markdown
    - if you find names of any person make them lower cased inside the query to fetch
    - when ever giving details of a timetable be mind full in including details like time slot,day,faculty name,room no
    - add interval of 1 if tomorrow and 2 if day after tomorrow
    - always limit max no.of slots retrieving to 2
    - if not da is mentioned assume it as today
    - force yourself to qreate a query based given examples and question dont answer with "We don't have related data in the database"

    consider below given eamples to understand the query formate 
    """, inputs=["table_info"])
    user_template = HumanMessagePromptTemplate.from_template(
        """
    
        user_Question: {input}
    
        your_answer:
        """, inputs=["input"])

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="few_shot_db", embedding_function=embedding_model)

    selector = SemanticSimilarityExampleSelector(vectorstore=vector_db, k=2)
    example_prompt = ChatPromptTemplate.from_messages([("human", "{user_question}"), ("ai", "{sql_query}")])
    few_shots = FewShotChatMessagePromptTemplate(
        example_selector=selector,
        example_prompt=example_prompt,
    )
    final_prompt = ChatPromptTemplate.from_messages([sys_template, few_shots, user_template])
    llm = GoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.2,
                             google_api_key=os.getenv("gemini_api"))
    DATABASE_URL = "mysql+pymysql://root:99230040570@localhost/faculty_meet"
    db = SQLDatabase.from_uri(DATABASE_URL)
    input_mapping = RunnableLambda(lambda x: {"input": x,
                                              "table_info": db.table_info})
    make_query = RunnableLambda(lambda response: get_result_from_sqldb(db, response.replace("```sql","").replace("```","")))
    out_sys_template = SystemMessagePromptTemplate.from_template("""
    Convert structured data into a natural language sentence following these rules:
    dont give any extra context or information lie this
    'Okay, here\'s the natural language conversion based on your rules and the provided data:\n\nGiven today is Thursday, the output is:\n\n
    Ensure clarity and correctness:

    The sentence must be concise, grammatically correct, and easy to understand.
    Convert 24-hour time format to 12-hour AM/PM format.

    Combine multiple entries based on the following cases:

    Same day, same room:
    Merge time slots into a single sentence using "and".
    Same day, different rooms:
    Merge entries using "and", keeping time slots in order.
    Different days:
    Group availability by day, ensuring proper chronological order.
    Join separate days using "and".
    Format the final response as:

    If availability is on one day:
    "Faculty_Name is available in room number X from A to B and from C to D on <Relative Day>."
    If availability is on multiple days:
    "Faculty_Name is available on <Relative Day> in room number X from A to B and in room number Y from C to D, and on <Next Relative Day> in room number Z from P to Q."


    """, inputs=[])
    out_chat_template = HumanMessagePromptTemplate.from_template("""
    dont give any other extra contents just give output alone
    Determine the correct day expression:                                        
    Fetch the current weekday automatically.
    If the given day matches today’s weekday, replace it with "today".
    If the given day is tomorrow, replace it with "tomorrow".
    If the given day is two days ahead, replace it with "day after tomorrow".
    if you recieve a empty set ansewr with "the faculty is free and He/She will be available in his cabin"
    Otherwise, keep the weekday name.

    here is the data
    {response}

    """, inputs=["response"])
    out_template = ChatPromptTemplate.from_messages([out_sys_template, out_chat_template])
    final_chain = (input_mapping
                   | final_prompt
                   | llm
                   | make_query
                   | out_template
                   | llm)
    fresponse = final_chain.invoke(input_query)
    return fresponse