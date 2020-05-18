from tkinter import messagebox
from tkinter import *
import smtplib, ssl
import numpy as np
from time import *
import datetime
import csv
import cv2
import os

# ----------Main Win-----------------------
root = Tk()
root.geometry("470x270")
root.resizable(width=False, height=False)
root.title("Welcome To APRS")

# module level variables 
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

# showSteps = True
showSteps = False

# ----------Set Path For Folder-----------------------
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


assure_path_exists("Details/")
assure_path_exists("KNN/")

# -----------Show win in center--------------
def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))


def fun(event):
    root.destroy()
    sleep(1)
    root1 = Tk()
    root1.geometry("450x250")
    root1.resizable(width=False, height=False)
    root1.title("Login")
    root1.focus_set()
    center(root1)

    # -----------Final Main Other Call---------------------

    def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
        p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # get 4 vertices of rotated rect

        cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)  # draw 4 red lines
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

    ###################################################################################################
    def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
        ptCenterOfTextAreaX = 0  # this will be the center of the area the text will be written to
        ptCenterOfTextAreaY = 0
        ptLowerLeftTextOriginX = 0  # this will be the bottom left of the area that the text will be written to
        ptLowerLeftTextOriginY = 0

        sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
        plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

        intFontFace = cv2.FONT_HERSHEY_SIMPLEX  # choose a plain jane font
        fltFontScale = float(plateHeight) / 30.0  # base font scale on height of plate area
        intFontThickness = int(round(fltFontScale * 1.5))  # base font thickness on font scale

        textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                             intFontThickness)  # call getTextSize

        # unpack roatated rect into center point, width and height, and angle
        ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
         fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

        intPlateCenterX = int(intPlateCenterX)  # make sure center is an integer
        intPlateCenterY = int(intPlateCenterY)

        ptCenterOfTextAreaX = int(intPlateCenterX)  # the horizontal location of the text area is the same as the plate

        if intPlateCenterY < (sceneHeight * 0.75):  # if the license plate is in the upper 3/4 of the image
            ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
                round(plateHeight * 1.6))  # write the chars in below the plate
        else:  # else if the license plate is in the lower 1/4 of the image
            ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
                round(plateHeight * 1.6))  # write the chars in above the plate
        # end if

        textSizeWidth, textSizeHeight = textSize  # unpack text size width and height

        ptLowerLeftTextOriginX = int(
            ptCenterOfTextAreaX - (textSizeWidth / 2))  # calculate the lower left origin of the text area
        ptLowerLeftTextOriginY = int(
            ptCenterOfTextAreaY + (textSizeHeight / 2))  # based on the text area center, width, and height

        # write the text on the image
        cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                    fltFontScale, SCALAR_YELLOW, intFontThickness)

    # -----------Final Main Call---------------------
    def final_main(x):
        import DetectChars
        import DetectPlates

        print("\nFinal Main")

        blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()  # attempt KNN training
        if blnKNNTrainingSuccessful == False:  # if KNN training was not successful
            print("\nerror: KNN traning was not successful\n")  # show error message
            return  # and exit program

        imgOriginalScene = cv2.imread("LicPlateImages/" + str(x) + ".png")

        if imgOriginalScene is None:  # if image was not read successfully
            print("\nerror: image not read from file \n\n")  # print error message to std out
            os.system("pause")  # pause so user can see error message
            return  # and exit program
        # end if

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)  # detect plates @@@@@ PY
        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  # detect chars in plates
        # cv2.imshow("imgOriginalScene", imgOriginalScene)

        if len(listOfPossiblePlates) == 0:  # if no plates were found
            print("\nno license plates were detected\n")  # inform user no plates were found
        else:  # else
            # if we get in here list of possible plates has at leat one plate

            # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
            listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

            # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
            licPlate = listOfPossiblePlates[0]

            cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
            cv2.imshow("imgThresh", licPlate.imgThresh)

            if len(licPlate.strChars) == 0:  # if no chars were found in the plate
                print("\nno characters were detected\n\n")  # show message
                return  # and exit program
            # end if

            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)  # draw red rectangle around plate

            print("\nlicense plate read from image = " + licPlate.strChars)  # write license plate text to std out

            NP = licPlate.strChars
            ts = time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%d/%m/%Y')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

            row = [date, timeStamp, NP]
            with open('Details/Car_Record.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()

            li = []
            with open('Details/Theft_Car.csv') as File:
                reader = csv.reader(File)
                for row in reader:
                    if (len(row) == 1):
                        li.append(row[0])

            if NP in li:
                print("\nTheft Car Found")

                # -------------Mail System------------------------------------------
                port = 587  # For starttls
                smtp_server = "smtp.gmail.com"
                sender_email = "itbpit2016@gmail.com"
                receiver_email = "avrbpit@gmail.com"
                password = "btech2016"
                message = "Subject: Alert Message Theft Car Detected\n\n\n" \
                          "Date: " + date + "\n" \
                          "Time: " + timeStamp + "\n" \
                          "Theft Car Number: " + NP + " \n\n\n" \
                          "This message is sent from Python Software APRS."

                context = ssl.create_default_context()
                try:
                    with smtplib.SMTP(smtp_server, port) as server:
                        server.ehlo()  # Can be omitted
                        server.starttls(context=context)
                        server.ehlo()  # Can be omitted
                        server.login(sender_email, password)
                        server.sendmail(sender_email, receiver_email, message)

                except Exception as e:
                    print(e)
                # ---------------------------------------------------------------------------------

                #messagebox.showwarning("Theft Car Detect", "Call Supervisor NOW.!..")
                print("\nAutomatic Mail System RUN")

            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)  # write license plate text on the image

            cv2.imwrite("Details/LASTimg.png", imgOriginalScene)  # write image out to file
            cv2.imshow("Details/LASTimg", imgOriginalScene)  # re-show scene image

        cv2.waitKey(0)
        staff_function_call()
        return

    # ----------staff_function_call------------------
    def staff_function_call():
        print("\nStaff Final Windows")
        try:
            root1.destroy()
        except Exception as e:
            print(e)

        root_M = Tk()
        root_M.geometry('470x430')
        root_M.resizable(False, False)
        root_M.config(background='#13a9f0')
        root_M.title("Staff Use")
        center(root_M)

        def disp():
            listNodes.delete(0, END)
            with open("Details/Car_Record.csv", 'r')as r:
                data = csv.reader(r)
                z1 = "   "
                z2 = "     "
                z3 = "      "
                for row in data:
                    if (len(row) == 3):
                        listNodes.insert(END, z1 + row[0] + z2 + row[1] + z3 + row[2])

        def dis():
            listNodes.delete(0, END)
            with open('Details/Theft_Car.csv') as File:
                reader = csv.reader(File)
                for row in reader:
                    if (len(row) == 1):
                        listNodes.insert(END, row[0])

        def st():
            x = variable.get()
            root_M.destroy()
            final_main(x)

        bg = "#13a9f0"

        def tick(time1=''):
            time2 = strftime('%I:%M:%S')
            if time2 != time1:
                l1.config(text=time2)
            l1.after(200, tick)

        Label(root_M, bg=bg).pack()
        f1 = Frame(root_M)
        f1.pack(side=TOP)
        l1 = Label(f1, font=('arial 15 bold'), bg=bg)
        l1.pack(side=RIGHT)
        y = strftime("%d/%m/%y     ")
        Label(f1, text=y, font=('arial 15 bold'), bg=bg).pack()
        tick()

        name = Label(root_M, text="Select Car Image", font=('arial 11 bold'), bg=bg)
        name.place(x=50, y=95)

        #################
        LF = os.listdir('LicPlateImages/')
        OptionList = []
        for i in LF:
            if (".png" == (os.path.splitext(i)[1]) or ".jpg" == (os.path.splitext(i)[1])):
                OptionList.append(int(os.path.splitext(i)[0]))
        OptionList.sort()

        variable = StringVar(root_M)
        variable.set(OptionList[0])
        opt = OptionMenu(root_M, variable, *OptionList)
        opt.config(font=('Helvetica', 12), bg=bg)
        opt.place(x=185, y=90)
        #################

        start = Button(root_M, text="Start", width=9, bd=3, height=2, bg='#13f01a', command=st)
        start.place(x=60, y=200)

        quit = Button(root_M, text="Exit", width=9, bd=3, height=2, bg="#e11b12", command=root_M.destroy)
        quit.place(x=180, y=200)

        Label(root_M, text="Theft List", font=('arial 11 bold'), bg=bg).place(x=325, y=90)

        frame = Frame(root_M)
        frame.place(x=320, y=120)
        listNodes = Listbox(frame, width=10, height=6, bd=4, font=('times'), selectbackground="blue",
                            activestyle="none")
        listNodes.pack(side="left", fill="y")
        scrollbar = Scrollbar(frame, orient="vertical")
        scrollbar.config(command=listNodes.yview)
        scrollbar.pack(side="right", fill="y")
        listNodes.config(yscrollcommand=scrollbar.set)
        dis()

        frame = Frame(root_M)
        frame.place(x=50, y=300)
        listNodes = Listbox(frame, width=35, height=4, bd=4, font=('times', 15, ' bold '), selectbackground="blue",
                            activestyle="none")
        listNodes.pack(side="left", fill="y")
        scrollbar = Scrollbar(frame, orient="vertical")
        scrollbar.config(command=listNodes.yview)
        scrollbar.pack(side="right", fill="y")
        listNodes.config(yscrollcommand=scrollbar.set)
        disp()

        show = "'Date'            'Time'              'Number'"
        lbl5 = Label(root_M, text=show, width=33, fg="#0055ff", bg="white", height=0, font=('times', 13, ' bold'))
        lbl5.place(x=55, y=305)

        root_M.mainloop()

    # --------staff call----------------------------------------------
    def staff_call(su, sp):
        with open("Details/Staff_Details.csv", 'r')as r:
            data = csv.reader(r)
            for row in data:
                if (len(row) == 2):
                    if (str(su) == row[0] and int(sp) == int(row[1])):
                        f = 1
                        break
                    else:
                        f = 0
            if (f == 1):
                staff_function_call()

            else:
                messagebox.showerror("Invalid Name", "Details Name Not Found.")

    # -------------------admin call knn-------------------------------
    def admin_call_knn():
        MIN_CONTOUR_AREA = 100
        RESIZED_IMAGE_WIDTH = 20
        RESIZED_IMAGE_HEIGHT = 30

        imgTrainingNumbers = cv2.imread("KNN/Training_Chars.png")  # read in training numbers image

        if imgTrainingNumbers is None:  # if image was not read successfully
            print("\nerror: image not read from file")  # print error message to std out
            quit()

        imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)  # get grayscale image
        imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

        # filter image from grayscale to black and white
        imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                          255,  # make pixels that pass the threshold full white
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          # use gaussian rather than mean, seems to give better results
                                          cv2.THRESH_BINARY_INV,
                                          # invert so foreground wihll be white, background will be black
                                          11,  # size of a pixel neighborhood used to calculate threshold value
                                          2)  # constant subtracted from the mean or weighted mean

        # cv2.imshow("imgThresh", imgThresh)      # show threshold image for reference

        imgThreshCopy = imgThresh.copy()  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

        npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                     # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                     cv2.RETR_EXTERNAL,  # retrieve the outermost contours only
                                                     cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points

        # declare empty numpy array, we will use this to write to file later
        # zero rows, enough cols to hold all image data
        npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        intClassifications = []  # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end

        # possible chars we are interested in are digits 0 through 9, put these in list intValidChars
        intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'),
                         ord('9'),
                         ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'),
                         ord('J'),
                         ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'),
                         ord('T'),
                         ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

        for npaContour in npaContours:  # for each contour
            if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:  # if contour is big enough to consider
                [intX, intY, intW, intH] = cv2.boundingRect(npaContour)  # get and break out bounding rect

                # draw rectangle around each contour as we ask user for input
                cv2.rectangle(imgTrainingNumbers,  # draw rectangle on original training image
                              (intX, intY),  # upper left corner
                              (intX + intW, intY + intH),  # lower right corner
                              (0, 0, 255),  # red
                              2)  # thickness

                imgROI = imgThresh[intY:intY + intH, intX:intX + intW]  # crop char out of threshold image
                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                                    RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage

                cv2.imshow("imgROIResized", imgROIResized)  # show resized image for reference
                # cv2.imshow("training_numbers.png",imgTrainingNumbers)  # show training numbers image, this will now have red rectangles drawn on it

                cv2.imwrite("KNN/Cache/R_imgTrainingNumbers.png", imgTrainingNumbers)

                img = cv2.imread('KNN/Cache/R_imgTrainingNumbers.png', cv2.IMREAD_UNCHANGED)
                scale_percent = 50
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                cv2.imshow("KNN/Cache/ResizedImage.png", resized)

                intChar = cv2.waitKey(0)  # get key press
                if intChar == 27:
                    cv2.destroyAllWindows()
                    admin_call_class()

                elif intChar in intValidChars:  # else if the char is in the list of chars we are looking for . . .

                    intClassifications.append(
                        intChar)  # append classification char to integer list of chars (we will convert to float later before writing to file)

                    npaFlattenedImage = imgROIResized.reshape((1,
                                                               RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                    npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage,
                                                   0)  # add current flattened impage numpy array to list of flattened image numpy arrays

        fltClassifications = np.array(intClassifications,
                                      np.float32)  # convert classifications list of ints to numpy array of floats

        npaClassifications = fltClassifications.reshape(
            (fltClassifications.size, 1))  # flatten numpy array of floats to 1d so we can write to file later

        print("\nTraining Complete")

        np.savetxt("KNN/Classifications.txt", npaClassifications)  # write flattened images to file
        np.savetxt("KNN/Flattened_images.txt", npaFlattenedImages)

    # -------------------admin add theft_car-------------------------------
    def theft_car():
        root_theft = Tk()
        root_theft.geometry('460x240')
        root_theft.resizable(False, False)
        root_theft.config(background='#13a9f0')
        root_theft.title("Add Theft Car")
        center(root_theft)

        def disp():
            listNodes.delete(0, END)
            with open('Details/Theft_Car.csv') as File:
                reader = csv.reader(File)
                for row in reader:
                    if (len(row) == 1):
                        listNodes.insert(END, row[0])

        def adT():
            u = No.get()

            if (len(u) == 0):
                messagebox.showinfo("Value Error", "Enter Number")

            else:
                print("\nAdd Theft Car")
                row = [u]
                with open('Details/Theft_Car.csv', 'a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()

                disp()
                No.delete(0, END)

        bg = "#13a9f0"
        heading = Label(root_theft, text="Theft Car List", font=('arial 15 bold'), bg=bg)
        heading.place(x=130, y=15)

        name = Label(root_theft, text="Car Number Plate: ", font=('arial 10'), bg=bg)
        name.place(x=40, y=90)

        No = Entry(root_theft, width=15, bd=3)
        No.place(x=170, y=90)
        No.focus_set()

        Add = Button(root_theft, text="Add", width=10, bg='#13f01a', command=adT)
        Add.place(x=50, y=170)

        quit = Button(root_theft, text="Exit", width=10, height=1, bg='#EE3D3D', command=root_theft.destroy)
        quit.place(x=170, y=170)

        frame = Frame(root_theft)
        frame.place(x=300, y=30)
        listNodes = Listbox(frame, width=12, height=8, bd=4, font=('times'), selectbackground="blue",
                            activestyle="none")
        listNodes.pack(side="left", fill="y")
        scrollbar = Scrollbar(frame, orient="vertical")
        scrollbar.config(command=listNodes.yview)
        scrollbar.pack(side="right", fill="y")
        listNodes.config(yscrollcommand=scrollbar.set)
        disp()

        root_theft.mainloop()

    # -------------------admin add staff-------------------------------

    def add_staff():
        root_sfa = Tk()
        root_sfa.geometry('460x240')
        root_sfa.resizable(False, False)
        root_sfa.config(background='#13a9f0')
        root_sfa.title("Add Staff")
        center(root_sfa)

        def dis():
            listNodes.delete(0, END)
            with open('Details/Staff_Details.csv', newline='') as File:
                reader = csv.reader(File)
                for row in reader:
                    if (len(row) == 2):
                        listNodes.insert(END, row[0] + "   " + row[1])

        def clr():
            Name.delete(0, END)
            PC.delete(0, END)

        def adS():
            u = Name.get()
            p = PC.get()

            if (len(u) == 0 and len(p) == 0):
                messagebox.showinfo("Value Error", "First Enter Staff Name")

            elif (len(u) == 0):
                messagebox.showinfo("Value Error", "First Enter Staff Name")

            elif (len(p) == 0):
                messagebox.showinfo("Value Error", "Enter Numeric PassCode")

            if (len(u) != 0 and len(p) != 0):
                try:
                    float(p)
                    u.isalpha()
                    print("\nStaff Add")
                    row = [u, p]
                    with open('Details/Staff_Details.csv', 'a+') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow(row)
                    csvFile.close()

                    dis()

                    Name.focus_set()
                    clr()

                except (TypeError, ValueError):
                    messagebox.showerror("Value Error", "Enter Only Valid Data Type")

        bg = "#13a9f0"
        heading = Label(root_sfa, text="Add Staff", font=('arial 15 bold'), bg=bg)
        heading.place(x=130, y=15)

        name = Label(root_sfa, text="Staff Name: ", bg=bg)
        pno = Label(root_sfa, text="Staff Password: ", bg=bg)

        name.place(x=50, y=70)
        pno.place(x=50, y=110)

        Name = Entry(root_sfa, width=15, bd=3)
        Name.place(x=170, y=70)
        Name.focus_set()
        PC = Entry(root_sfa, width=15, bd=3)
        PC.place(x=170, y=110)

        Add = Button(root_sfa, text="Add", width=8, bg='#13f01a', command=adS)
        Add.place(x=50, y=170)

        quit = Button(root_sfa, text="Clear", width=7, height=1, bg='#EE3D3D', command=clr)
        quit.place(x=130, y=170)

        quit = Button(root_sfa, text="Exit", width=7, height=1, bg='#EE3D3D', command=root_sfa.destroy)
        quit.place(x=200, y=170)

        frame = Frame(root_sfa)
        frame.place(x=300, y=30)
        listNodes = Listbox(frame, width=12, height=8, bd=4, font=('times'), selectbackground="blue",
                            activestyle="none")
        listNodes.pack(side="left", fill="y")
        scrollbar = Scrollbar(frame, orient="vertical")
        scrollbar.config(command=listNodes.yview)
        scrollbar.pack(side="right", fill="y")
        listNodes.config(yscrollcommand=scrollbar.set)
        dis()

        root_sfa.mainloop()

    # ------------------admin_call_class------------------------------

    def admin_call_class():
        root1.destroy()
        root2 = Tk()
        root2.geometry('580x330')
        # root2.config(background='#108ff2')
        root2.resizable(False, False)
        root2.title("Welcome Admin")
        center(root2)

        def tick(time1=''):
            time2 = strftime('%I:%M:%S')
            if time2 != time1:
                l1.config(text=time2)
            l1.after(200, tick)

        Label(root2).pack()
        f1 = Frame(root2)
        f1.pack(side=TOP)
        l1 = Label(f1, font=('arial 15 bold'))
        l1.pack(side=RIGHT)
        y = strftime("%d/%m/%y         ")
        Label(f1, text=y, font=('arial 15 bold')).pack()
        tick()

        bg = "#6ce621"
        b1 = Button(root2, text="ADD No. Plate", font=('arial 10'), bg=bg, command=theft_car, borderwidth=4,
                    relief=RAISED, width=12,
                    height=2)
        b1.place(x=60, y=130)
        b3 = Button(root2, text="ADD Satff", font=('arial 10'), bg=bg, command=add_staff, borderwidth=4,
                    relief=RAISED,
                    width=12, height=2)
        b3.place(x=290, y=130)
        b4 = Button(root2, text="KNN Learning", font=('arial 10'), bg=bg, command=admin_call_knn, borderwidth=4,
                    relief=RAISED, width=12,
                    height=2)
        b4.place(x=170, y=220)
        b6 = Button(root2, text="Exit", font=('arial 10'), bg="#e11b12", command=quit, borderwidth=4, relief=RAISED,
                    width=12, height=2)
        b6.place(x=410, y=220)

        root2.mainloop()

    # ---------clr scr------------------------------------------
    def clr():
        e1.delete(0, END)
        e2.delete(0, END)

    # -------- see Radiobutton & login--------------------------------------
    def login():

        u = (e1.get())
        p = (e2.get())

        if (len(u) == 0 and len(p) == 0):
            messagebox.showinfo("Value Error", "First Enter User ID")

        elif (len(u) == 0):
            messagebox.showinfo("Value Error", "First Enter User ID")

        elif (len(p) == 0):
            messagebox.showinfo("Value Error", "Enter Numeric PassCode")

        ch = selected.get()

        # -----------------------staff log in
        if (ch == 1):

            if (len(u) != 0 and len(p) != 0):
                try:
                    float(p)
                    u.isalpha()
                    staff_call(u, p)

                except (TypeError, ValueError):
                    messagebox.showerror("Value Error", "Enter Only Valid Data Type----")

        else:  # ---------------------------------Admin login Password
            au = "q"
            ap = 1

            if (len(u) != 0 and len(p) != 0):
                try:
                    float(p)
                    u.isalpha()
                    if (au == str(u) and ap == int(p)):
                        print("\nWelcome Admin")
                        admin_call_class()

                    else:
                        messagebox.showerror("Value Error", "Input Data is Invalide")

                except (TypeError, ValueError):
                    messagebox.showerror("Value Error", "Enter Only Valid Data Type")

    # --------Radiobutton--------------------------------------------
    selected = IntVar()
    rad1 = Radiobutton(root1, text='Staff', value=1, variable=selected)
    rad2 = Radiobutton(root1, text='Admin', value=2, variable=selected)
    selected.set(1)
    rad1.place(x=200, y=30)
    rad2.place(x=270, y=30)

    # ----------------------------------------------------------------
    l = Label(root1, text="Login Options: ")
    l.place(x=80, y=30)

    l1 = Label(root1, text="Enter Username: ")
    l2 = Label(root1, text="Enter Password: ")
    l1.place(x=100, y=85)
    l2.place(x=100, y=120)

    e1 = Entry(root1, bd=3)
    e2 = Entry(root1, bd=3, show="*")
    e1.place(x=230, y=90)
    e1.focus_set()
    e2.place(x=230, y=120)

    b1 = Button(root1, text="Login", width=10, height=1, bd=3, bg='#1DC550', command=login)
    b1.place(x=80, y=180)
    b2 = Button(root1, text="Clear", width=10, height=1, bd=3, command=clr)
    b2.place(x=190, y=180)
    b3 = Button(root1, text="Exit", width=10, height=1, bd=3, bg='#EE3D3D', command=quit)
    b3.place(x=300, y=180)

    root1.mainloop()


center(root)

root.bind("<Button-1>", fun)
root.bind("<Return>", fun)
root.bind("<Tab>", fun)
root.bind("<space>", fun)

l1 = Label(root, text="Automatic Plate\nRecognition System", font=('arial 24 bold'), fg="#009bff")
l1.pack(fill="both", expand=True)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(1, weight=1)
l2 = Label(root, text="by", font=('arial 12'))
l2.pack(fill="both", expand=True)
l3 = Label(root, text="Anand Vijay Rajsri (00120807717)", font=('arial 18'), fg="#009bff")
l3.pack(fill="both", expand=True)
l4 = Label(root, text="... Tap To Continue ...", relief=SUNKEN, bd=1)
l4.pack(side=BOTTOM, fill=X)

root.mainloop()
