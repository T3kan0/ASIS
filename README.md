# ASIS
Tekano Mbonani

## System Docs 📃
An automated, user-interactive tutorial attendance impact PDF report generator. Using a combination of ***R*** and ***python*** languages, the code performs hypothesis testing and linear regression/correlation analysis between student tutorial attendance and their final exam marks, to determine the tutorial impact on students. The code requires student tutorial attedance and performance data files to compile a tutorial impact report per faculty, term (semester) and campus. The code is user-interactive, the user must know the faculty, term and campus they want to report on. To achieve this, the data is combined to determine student attendance frequency per module (subject) for the selected faculty, term and campus. The majority of the ***python*** code is modularized through functions to get a clean code, therefore, the user needs to run only one of these. I wrote these codes for a large tutorial project for students of the University of Free State, comprised of seven academic faculties on three learning campuses. The code allowed my clients to assess the project's performance and impact on its attendees, in real-time, within minutes.

## Software Requirements 🔌
You will need to install the following software on your system in order to run/edit the Python and R scripts.
* Mac OS/ Ubuntu 18.04 OS
* Python 3.10.12
* R 4.3.1
* Textedit/ IDE - spyder, jupyter-notebook or R-studio
* libraries
  * pandas
  * numpy
  * scipy
  * pyreadr
  * datetime
  * fpdf
  * matplotlib
  * seaborn
  * glob
  * dplyr

### About the Data 💾 
The data used here was collected between 2019 and the first term of 2022, by the tutorial project of students on the University of Free State. The data was collected and stored weekly, where I have access to perform statistical analysis and reporting.
