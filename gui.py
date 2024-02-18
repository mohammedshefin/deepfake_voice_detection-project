from tkinter import *
from PIL import ImageTk, Image  # type "Pip install pillow" in your terminal to install ImageTk and Image module
import sqlite3
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import cv2 as cv
import cv2
import numpy as np
import time
import os
import librosa
import librosa.display
import pickle
from tensorflow.keras.models import load_model

con = sqlite3.connect('userdata.db')
cur = con.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS record(
                    name text, 
                    email text, 
                    password text
                )
            ''')
con.commit()


window = Tk()
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)
window.state('zoomed')
window.resizable(0, 0)
window.title('DeepFake Voice Detection')


#loading standardscaler
scaler=pickle.load(open('Project_Saved_Models/scaler.pkl','rb'))
#load the trained model 
loaded_model=load_model('Project_Saved_Models/cnn_lstm_model.h5')

# Window Icon Photo
icon = PhotoImage(file='images\\pic-icon.png')
window.iconphoto(True, icon)

LoginPage = Frame(window)
RegistrationPage = Frame(window)
HomePage = Frame(window)


for frame in (LoginPage, RegistrationPage):
    frame.grid(row=0, column=0, sticky='nsew')


for frame in (LoginPage, HomePage):
    frame.grid(row=0, column=0, sticky='nsew')

def show_frame(frame):
    frame.tkraise()

def show_frame1(frame):
    frame.tkraise()
    my_label.config(text="")
    label.config(text="")
    list_box.delete(0,END)

show_frame(LoginPage)



#extract features
def _extract__Features_(_data,sample_rate):
    # Zeor crossing rate
    _result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=_data).T, axis=0)
    _result=np.hstack((_result, zcr)) 

    # Chroma
    stft = np.abs(librosa.stft(_data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    _result = np.hstack((_result, chroma_stft)) 

   #mfcc
    mfcc = np.mean(librosa.feature.mfcc(y=_data, sr=sample_rate).T, axis=0)
    _result = np.hstack((_result, mfcc)) 

    # RMS Value
    rms = np.mean(librosa.feature.rms(y=_data).T, axis=0)
    _result = np.hstack((_result, rms))

   #melspectogram
    mel = np.mean(librosa.feature.melspectrogram(y=_data, sr=sample_rate).T, axis=0)
    _result = np.hstack((_result, mel)) # stacking horizontally
    
    return _result

def _get__Features_(path):
    
    _data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    print(path)
    
   
    res1 = _extract__Features_(_data,sample_rate)
    _result = np.array(res1)

    
    return _result


def prediction():
    
    list_box.insert(1,"Loading Audio")
    list_box.insert(2,"")
    list_box.insert(3,"Feature Extraction")
    list_box.insert(4,"")
    list_box.insert(5,"Loading CNN-LSTM model")
    list_box.insert(6,"")
    list_box.insert(7,"Prediction")

    feat=_get__Features_(path)
    feat=np.array([feat])
    #perform standardization
    feat=scaler.transform(feat)
    #expand dimension
    feat = np.expand_dims(feat, axis=2)
   
    #prediction using the model
    pred=loaded_model.predict(feat)[0]
    pred=pred[0]
    print("PRED : ",pred)

    if pred<=0.5:
        print("Human Voice")
        a="Human Voice"
    if pred>0.5:
        print("Deepfake Voice")
        a="Deepfake Voice"

    messagebox.showinfo('Result', a)

def Upload():
    global path,my_label
    label.config(text='No files uploaded',foreground="red",font="arial 10")
    my_label.config(text='')
    list_box.delete(0,END)
    path=askopenfilename(title='Open a file',
                         initialdir='Test',
                         filetypes=[("WAV","*.wav")])
    print("<<<<<<<<<<<<<",path)
    if(path==''):
        label.config(text="No files uploaded",foreground="red",font="arial 10")
    else:
        global my_file_name
        my_file_name=os.path.basename(path)

        my_label.config(text=my_file_name)
        my_label.place(x=240,y=280)
        label.config(text="Audio Uploaded",font="arial 12", foreground="green")


design_frame10 = Listbox(HomePage, bg='#0c71b9', width=115, height=50, highlightthickness=0, borderwidth=0)
design_frame10.place(x=0, y=0)

design_frame20 = Listbox(HomePage, bg='#1e85d0', width=115, height=50, highlightthickness=0, borderwidth=0)
design_frame20.place(x=676, y=0)

design_frame30 = Listbox(HomePage, bg='#f8f8f8', width=100, height=33, highlightthickness=0, borderwidth=0)
design_frame30.place(x=75, y=106)

design_frame40 = Listbox(HomePage, bg='#61a9de', width=100, height=33, highlightthickness=0, borderwidth=0)
design_frame40.place(x=676, y=106)

upload_pic_button=Button(design_frame30,text="Upload Audio",command=Upload,bg="pink")
upload_pic_button.place(x=250,y=180)

global my_label
my_label=Label(design_frame30,bg="#f8f8f8")
global label
label=Label(design_frame30,text="No files uploaded",foreground="red",font="arial 10",bg="#f8f8f8")
label.place(x=240,y=220)


predict_button=Button(design_frame30,text="Predict",command=prediction,bg="light green")
predict_button.place(x=270,y=350)

refresh_button=Button(design_frame30,text="Refresh",command=lambda: show_frame1(HomePage),bg="orange")
refresh_button.place(x=270,y=400)

name_label=Label(design_frame40,text="Process",font="arial 14",bg="#61a9de",fg="white")
name_label.place(x=250,y=100)

global list_box
list_box=Listbox(design_frame40,height=12,width=31)
list_box.place(x=200,y=170)


# =====================================================================================================================
# =====================================================================================================================
# ==================== LOGIN PAGE =====================================================================================
# =====================================================================================================================
# =====================================================================================================================
#######################################################################################################################
def login_response(e1,p1):
    global nm
    try:
        con = sqlite3.connect('userdata.db')
        c = con.cursor()
        for row in c.execute("Select * from record"):
            name = row[0]
            em = row[1]
            pw=row[2]
        
    except Exception as ep:
        messagebox.showerror('', ep)

    uname = e1.get()
    upwd = p1.get()
    check_counter=0
    if uname == "":
       warn = "Username can't be empty"
    else:
        check_counter += 1
    if upwd == "":
        warn = "Password can't be empty"
    else:
        check_counter += 1
    if check_counter == 2:
        if (uname == em and upwd == pw):
            messagebox.showinfo('Login Status', 'Logged in Successfully!')
            show_frame(HomePage)
            # home()

        else:
            messagebox.showerror('Login Status', 'invalid username or password')
    else:
        messagebox.showerror('', warn)
#######################################################################################################################
design_frame1 = Listbox(LoginPage, bg='#0c71b9', width=115, height=50, highlightthickness=0, borderwidth=0)
design_frame1.place(x=0, y=0)

design_frame2 = Listbox(LoginPage, bg='#1e85d0', width=115, height=50, highlightthickness=0, borderwidth=0)
design_frame2.place(x=676, y=0)

design_frame3 = Listbox(LoginPage, bg='#1e85d0', width=100, height=33, highlightthickness=0, borderwidth=0)
design_frame3.place(x=75, y=106)

design_frame4 = Listbox(LoginPage, bg='#f8f8f8', width=100, height=33, highlightthickness=0, borderwidth=0)
design_frame4.place(x=676, y=106)

# ====== Email ====================
email_entry = Entry(design_frame4, fg="#a7a7a7", font=("yu gothic ui semibold", 12), highlightthickness=2)
email_entry.place(x=134, y=170, width=256, height=34)
email_entry.config(highlightbackground="black", highlightcolor="black")
email_label = Label(design_frame4, text='• Email account', fg="#89898b", bg='#f8f8f8', font=("yu gothic ui", 11, 'bold'))
email_label.place(x=130, y=140)

# ==== Password ==================
password_entry1 = Entry(design_frame4, fg="#a7a7a7", font=("yu gothic ui semibold", 12), show='•', highlightthickness=2)
password_entry1.place(x=134, y=250, width=256, height=34)
password_entry1.config(highlightbackground="black", highlightcolor="black")
password_label = Label(design_frame4, text='• Password', fg="#89898b", bg='#f8f8f8', font=("yu gothic ui", 11, 'bold'))
password_label.place(x=130, y=220)





# function for show and hide password
def password_command():
    if password_entry1.cget('show') == '•':
        password_entry1.config(show='')
    else:
        password_entry1.config(show='•')


# ====== checkbutton ==============
checkButton = Checkbutton(design_frame4, bg='#f8f8f8', command=password_command, text='show password')
checkButton.place(x=140, y=288)

# ========= Buttons ===============
SignUp_button = Button(LoginPage, text='Sign up', font=("yu gothic ui bold", 12), bg='#f8f8f8', fg="#89898b",
                       command=lambda: show_frame(RegistrationPage), borderwidth=0, activebackground='#1b87d2', cursor='hand2')
SignUp_button.place(x=1100, y=175)

# ===== Welcome Label ==============
welcome_label = Label(design_frame4, text='Welcome', font=('Arial', 20, 'bold'), bg='#f8f8f8')
welcome_label.place(x=130, y=15)

# ======= top Login Button =========
login_button = Button(LoginPage, text='Login', font=("yu gothic ui bold", 12), bg='#f8f8f8', fg="#89898b",
                      borderwidth=0, activebackground='#1b87d2', cursor='hand2')
login_button.place(x=845, y=175)

login_line = Canvas(LoginPage, width=60, height=5, bg='#1b87d2')
login_line.place(x=840, y=203)

# ==== LOGIN  down button ============
loginBtn1 = Button(design_frame4, fg='#f8f8f8', text='Login', bg='#1b87d2', font=("yu gothic ui bold", 15),
                   cursor='hand2', activebackground='#1b87d2',command=lambda: login_response(email_entry,password_entry1))
loginBtn1.place(x=133, y=340, width=256, height=50)


# ======= ICONS =================

# ===== Email icon =========
email_icon = Image.open('images\\email-icon.png')
photo = ImageTk.PhotoImage(email_icon)
emailIcon_label = Label(design_frame4, image=photo, bg='#f8f8f8')
emailIcon_label.image = photo
emailIcon_label.place(x=105, y=174)

# ===== password icon =========
password_icon = Image.open('images\\pass-icon.png')
photo = ImageTk.PhotoImage(password_icon)
password_icon_label = Label(design_frame4, image=photo, bg='#f8f8f8')
password_icon_label.image = photo
password_icon_label.place(x=105, y=254)

# ===== picture icon =========
picture_icon = Image.open('images\\pic-icon.png')
photo = ImageTk.PhotoImage(picture_icon)
picture_icon_label = Label(design_frame4, image=photo, bg='#f8f8f8')
picture_icon_label.image = photo
picture_icon_label.place(x=280, y=5)

# ===== Left Side Picture ============
side_image = Image.open('images\\vector.png')
photo = ImageTk.PhotoImage(side_image)
side_image_label = Label(design_frame3, image=photo, bg='#1e85d0')
side_image_label.image = photo
side_image_label.place(x=50, y=10)


# ===================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# ==================== REGISTRATION PAGE ==============================================================================
# =====================================================================================================================
# =====================================================================================================================
########################################################################################################################
def insert_record(e1n,e2e,e3p,e4cp):
    warn = ""
    if e1n.get() == "":
       warn = "Name can't be empty"
       messagebox.showerror('Error', warn)        
    elif e2e.get() == "":
        warn = "Email can't be empty"
        messagebox.showerror('Error', warn)   
    elif e3p.get() == "":
       warn = "Password can't be empty"
       messagebox.showerror('Error', warn)
    elif e4cp.get() == "":
        warn = "Re-enter password can't be empty"
        messagebox.showerror('Error', warn)
    elif e3p.get() != e4cp.get():
        warn = "Passwords didn't match!"
        messagebox.showerror('Error', warn)
    else:
        try:
            con = sqlite3.connect('userdata.db')
            cur = con.cursor()
            cur.execute("INSERT INTO record VALUES (:name, :email, :password)", {
                            'name': e1n.get(),
                            'email': e2e.get(),
                            'password': e3p.get()

            })
            con.commit()
            messagebox.showinfo('confirmation', 'Registered Successfully')
            show_frame(LoginPage)
        except Exception as ep:
            messagebox.showerror('', ep) 

########################################################################################################################
design_frame5 = Listbox(RegistrationPage, bg='#0c71b9', width=115, height=50, highlightthickness=0, borderwidth=0)
design_frame5.place(x=0, y=0)

design_frame6 = Listbox(RegistrationPage, bg='#1e85d0', width=115, height=50, highlightthickness=0, borderwidth=0)
design_frame6.place(x=676, y=0)

design_frame7 = Listbox(RegistrationPage, bg='#1e85d0', width=100, height=33, highlightthickness=0, borderwidth=0)
design_frame7.place(x=75, y=106)

design_frame8 = Listbox(RegistrationPage, bg='#f8f8f8', width=100, height=33, highlightthickness=0, borderwidth=0)
design_frame8.place(x=676, y=106)

# ==== Full Name =======
name_entry = Entry(design_frame8, fg="#a7a7a7", font=("yu gothic ui semibold", 12), highlightthickness=2)
name_entry.place(x=284, y=150, width=286, height=34)
name_entry.config(highlightbackground="black", highlightcolor="black")
name_label = Label(design_frame8, text='•Full Name', fg="#89898b", bg='#f8f8f8', font=("yu gothic ui", 11, 'bold'))
name_label.place(x=280, y=120)

# ======= Email ===========
email_entry1 = Entry(design_frame8, fg="#a7a7a7", font=("yu gothic ui semibold", 12), highlightthickness=2)
email_entry1.place(x=284, y=220, width=286, height=34)
email_entry1.config(highlightbackground="black", highlightcolor="black")
email_label = Label(design_frame8, text='•Email', fg="#89898b", bg='#f8f8f8', font=("yu gothic ui", 11, 'bold'))
email_label.place(x=280, y=190)

# ====== Password =========
password_entry = Entry(design_frame8, fg="#a7a7a7", font=("yu gothic ui semibold", 12), show='•', highlightthickness=2)
password_entry.place(x=284, y=295, width=286, height=34)
password_entry.config(highlightbackground="black", highlightcolor="black")
password_label = Label(design_frame8, text='• Password', fg="#89898b", bg='#f8f8f8',
                       font=("yu gothic ui", 11, 'bold'))
password_label.place(x=280, y=265)


def password_command2():
    if password_entry.cget('show') == '•':
        password_entry.config(show='')
    else:
        password_entry.config(show='•')


checkButton = Checkbutton(design_frame8, bg='#f8f8f8', command=password_command2, text='show password')
checkButton.place(x=290, y=330)


# ====== Confirm Password =============
confirmPassword_entry1 = Entry(design_frame8, fg="#a7a7a7", font=("yu gothic ui semibold", 12), highlightthickness=2)
confirmPassword_entry1.place(x=284, y=385, width=286, height=34)
confirmPassword_entry1.config(highlightbackground="black", highlightcolor="black")
confirmPassword_label = Label(design_frame8, text='• Confirm Password', fg="#89898b", bg='#f8f8f8',
                              font=("yu gothic ui", 11, 'bold'))
confirmPassword_label.place(x=280, y=355)

# ========= Buttons ====================
SignUp_button = Button(RegistrationPage, text='Sign up', font=("yu gothic ui bold", 12), bg='#f8f8f8', fg="#89898b",
                       command=lambda: show_frame(RegistrationPage), borderwidth=0, activebackground='#1b87d2', cursor='hand2')
SignUp_button.place(x=1100, y=175)

SignUp_line = Canvas(RegistrationPage, width=60, height=5, bg='#1b87d2')
SignUp_line.place(x=1100, y=203)

# ===== Welcome Label ==================
welcome_label = Label(design_frame8, text='Welcome', font=('Arial', 20, 'bold'), bg='#f8f8f8')
welcome_label.place(x=130, y=15)

# ========= Login Button =========
login_button = Button(RegistrationPage, text='Login', font=("yu gothic ui bold", 12), bg='#f8f8f8', fg="#89898b",
                      borderwidth=0, activebackground='#1b87d2', command=lambda: show_frame(LoginPage), cursor='hand2')
login_button.place(x=845, y=175)

# ==== SIGN UP down button ============
signUp2 = Button(design_frame8, fg='#f8f8f8', text='Sign Up', bg='#1b87d2', font=("yu gothic ui bold", 15),
                 cursor='hand2', activebackground='#1b87d2',command=lambda: insert_record(name_entry,email_entry1,password_entry,confirmPassword_entry1))
signUp2.place(x=285, y=435, width=286, height=50)

# ===== password icon =========
password_icon = Image.open('images\\pass-icon.png')
photo = ImageTk.PhotoImage(password_icon)
password_icon_label = Label(design_frame8, image=photo, bg='#f8f8f8')
password_icon_label.image = photo
password_icon_label.place(x=255, y=300)

# ===== confirm password icon =========
confirmPassword_icon = Image.open('images\\pass-icon.png')
photo = ImageTk.PhotoImage(confirmPassword_icon)
confirmPassword_icon_label = Label(design_frame8, image=photo, bg='#f8f8f8')
confirmPassword_icon_label.image = photo
confirmPassword_icon_label.place(x=255, y=390)

# ===== Email icon =========
email_icon = Image.open('images\\email-icon.png')
photo = ImageTk.PhotoImage(email_icon)
emailIcon_label = Label(design_frame8, image=photo, bg='#f8f8f8')
emailIcon_label.image = photo
emailIcon_label.place(x=255, y=225)

# ===== Full Name icon =========
name_icon = Image.open('images\\name-icon.png')
photo = ImageTk.PhotoImage(name_icon)
nameIcon_label = Label(design_frame8, image=photo, bg='#f8f8f8')
nameIcon_label.image = photo
nameIcon_label.place(x=252, y=153)

# ===== picture icon =========
picture_icon = Image.open('images\\pic-icon.png')
photo = ImageTk.PhotoImage(picture_icon)
picture_icon_label = Label(design_frame8, image=photo, bg='#f8f8f8')
picture_icon_label.image = photo
picture_icon_label.place(x=280, y=5)

# ===== Left Side Picture ============
side_image = Image.open('images\\vector.png')
photo = ImageTk.PhotoImage(side_image)
side_image_label = Label(design_frame7, image=photo, bg='#1e85d0')
side_image_label.image = photo
side_image_label.place(x=50, y=10)


window.mainloop()
