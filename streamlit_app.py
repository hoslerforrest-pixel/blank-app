import io
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(page_title="Mentee-Mentor pairing", page_icon="🤝", layout="wide")

# ==========================================
# LOGIN SCREEN
# ==========================================
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "WreckEm":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Hides the password from session state for security
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 Admin Login Required")
        st.text_input("Please enter the password to access the matchmaker:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 Admin Login Required")
        st.text_input("Please enter the password to access the matchmaker:", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect. Please try again.")
        return False
    else:
        return True

# ==========================================
# MAIN APP (Only runs if login is successful)
# ==========================================
if check_password():
    st.title("Mentee-Mentor pairing")
    st.write("Upload the SurveyMonkey export files below to calculate optimal pairings.")

    col1, col2 = st.columns(2)
    with col1:
        mentee_file = st.file_uploader("Upload Mentee Survey (.xlsx)", type=["xlsx"])
    with col2:
        mentor_file = st.file_uploader("Upload Mentor Survey (.xlsx)", type=["xlsx"])

    def calculate_score(mentee, mentor):
        score = 0

        # 1. Core Alignment: Major Match (Baseline 50 points)
        if str(mentee["Major"]).strip().lower() == str(mentor["Major"]).strip().lower():
            score += 50
            # Preference Boost (Extra 30 points if it is a priority)
            if mentee["Pref_Major"]:
                score += 30

        # 2. Soft Skills: Hometown Match (Baseline 10 points)
        mentee_town = str(mentee["Hometown"]).strip().lower()
        mentor_town = str(mentor["Hometown"]).strip().lower()
        
        # Ensure neither field is completely blank before comparing
        if mentee_town and mentor_town and mentee_town != "nan" and mentor_town != "nan":
            if mentee_town in mentor_town or mentor_town in mentee_town:
                score += 10
                # Preference Boost (Extra 20 points if it is a priority)
                if mentee["Pref_Hometown"]:
                    score += 20

        # 3. Soft Skills: Hobbies NLP Match (Up to 10 points)
        mentee_hobbies = str(mentee["Hobbies"])
        mentor_hobbies = str(mentor["Hobbies"])
        
        # Ensure neither field is completely blank before processing
        if mentee_hobbies != "nan" and mentor_hobbies != "nan":
            try:
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf = vectorizer.fit_transform([mentee_hobbies, mentor_hobbies])
                sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                
                score += sim * 10
                # Preference Boost (Extra 20 points if it is a priority)
                if mentee["Pref_Hobbies"]:
                    score += sim * 20
            except ValueError:
                pass

        # 4. The Anti-Zero Penalty
        # Forces the algorithm to completely avoid pairings with zero points
        if score <= 0:
            score = -500

        return score

    if st.button("Generate Matches", type="primary"):
        if mentee_file and mentor_file:
            with st.spinner("Processing surveys and running optimization..."):
                # Load raw data
                mentees_raw = pd.read_excel(mentee_file, header=[0, 1])
                mentors_raw = pd.read_excel(mentor_file, header=[0, 1])

                # Clean mentees
                clean_mentees = pd.DataFrame({
                    "Email": mentees_raw[("TTU Email Address", "Open-Ended Response")],
                    "Major": mentees_raw[("What is your intended engineering major?", "Response")],
                    "Hometown": mentees_raw[("Where are you from?", "Open-Ended Response")],
                    "Hobbies": mentees_raw[("What are your interests or hobbies?", "Open-Ended Response")],
                    "Pref_Major": mentees_raw[("What would you prefer to have in common with your mentor?Select all that apply", "Engineering major")].notna(),
                    "Pref_Hometown": mentees_raw[("What would you prefer to have in common with your mentor?Select all that apply", "Hometown/region")].notna(),
                    "Pref_Hobbies": mentees_raw[("What would you prefer to have in common with your mentor?Select all that apply", "Hobbies/interests")].notna(),
                }).dropna(subset=["Email"]).reset_index(drop=True)

                # Clean mentors
                clean_mentors = pd.DataFrame({
                    "Email": mentors_raw[("TTU Email Address", "Open-Ended Response")],
                    "Major": mentors_raw[("Major (Specific Engineering Major)", "Open-Ended Response")],
                    "Hometown": mentors_raw[("Home Town", "Open-Ended Response")],
                    "Hobbies": mentors_raw[("What are your hobbies outside of academics? What do you do for fun?", "Open-Ended Response")],
                }).dropna(subset=["Email"]).reset_index(drop=True)

                # Build matrix
                num_mentees = len(clean_mentees)
                num_mentors = len(clean_mentors)
                score_matrix = np.zeros((num_mentees, num_mentors))

                for i in range(num_mentees):
                    for j in range(num_mentors):
                        score_matrix[i, j] = calculate_score(clean_mentees.iloc[i], clean_mentors.iloc[j])

                # Hungarian algorithm
                cost_matrix = -1 * score_matrix
                mentee_idx, mentor_idx = linear_sum_assignment(cost_matrix)

                # Assemble results
                matches = []
                for m_idx, men_idx in zip(mentee_idx, mentor_idx):
                    matches.append({
                        "Mentee Email": clean_mentees.iloc[m_idx]["Email"],
                        "Mentee Major": clean_mentees.iloc[m_idx]["Major"],
                        "Mentor Email": clean_mentors.iloc[men_idx]["Email"],
                        "Mentor Major": clean_mentors.iloc[men_idx]["Major"],
                        "Compatibility Score": round(score_matrix[m_idx, men_idx], 2),
                    })

                results_df = pd.DataFrame(matches)

                # Create in-memory Excel file for download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    results_df.to_excel(writer, index=False, sheet_name="Matches")
                excel_data = output.getvalue()

                st.success(f"Successfully matched {len(results_df)} pairs!")

                # Download button
                st.download_button(
                    label="📥 Download Roster (Excel)",
                    data=excel_data,
                    file_name="Mentee-Mentor_Final_Matches.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                # Preview on screen
                st.subheader("Match Preview")
                st.dataframe(results_df, use_container_width=True)
        else:
            st.warning("Please upload both spreadsheets before generating matches.")
