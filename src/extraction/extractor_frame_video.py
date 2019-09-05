
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Extract frames from video

    Selecter folder where stay videos, before select folder where extract frames.

    Name: extractor_frame_video.py
    Author: Diego Andre Sant Ana  ( diego.santana@ifms.edu.br  )
"""
try:
    # for Python2
    import Tkinter as Tk
    import tkFileDialog
    from Tkinter import *
except ImportError:
    # for Python3
    import tkinter as Tk
    from tkinter import filedialog as tkFileDialog
    from tkinter import *
import os
from os import listdir
from os.path import isfile, join
import cv2
import threading


#
from util.utils import TimeUtils




class ExtractFM(object):

    def __init__(self):

        pass
    #Method Run, you can passa tkParent to manipulate objet to tk parent, this code use to console to print process extract frame movies.
    def run(self,tkParent):

        self.tk=tkParent
        self.folder_a=None
        self.folder_b=None
        #window
        self.window = Toplevel()

        self.window.title("Extract frames from videos")
        self.window.attributes('-zoomed', False)
        self.window.geometry("450x275+300+300")
        self.frame = Frame(self.window )
        self.frame.pack(fill=X, expand=False)
        #find a folder with videos
        self.folder_source = Button(self.frame, text="Select source folder", padx=5, pady=5, command = self.select_folder_source)
        self.folder_source.pack(padx=5, pady=5, expand=True, fill=X)
        #ToolTip(self.folder_source, "Select folder with videos from extract frames!")

        self.label_a =Label(self.frame, text="Folder no selected", padx=5, pady=5)
        self.label_a.pack(padx=5, pady=5, expand=True, fill=X)


        self.folder_export = Button(self.frame, text="Select destination folder", padx=5, pady=5, command = self.select_folder_export)
        self.folder_export.pack(padx=5, pady=5, expand=True, fill=X)
        #ToolTip(self.folder_export , "Select folder to save frames!")

        self.label_b =Label(self.frame, text="Folder no selected", padx=5, pady=5)
        self.label_b.pack(padx=5, pady=5, expand=True, fill=X)

        self.label_c =Label(self.frame, text="Each of the N frames draws a frame ", padx=5, pady=5)
        self.label_c.pack(padx=5, pady=5, expand=True, fill=X)
        #number de frame by extract one picture
        self.val_frame=IntVar()
        self.val_frame.set(30)

        self.spinbox = Spinbox(self.frame, from_=1.0, to=100.0, textvariable=self.val_frame)

        self.spinbox.configure(activebackground="#f9f9f9")
        self.spinbox.configure(background="white")

        self.spinbox.configure(buttonbackground="wheat")
        self.spinbox.configure(disabledforeground="#b8a786")

        self.spinbox.configure(highlightbackground="black")
        self.spinbox.configure(selectbackground="#c4c4c4")

        self.spinbox.pack(padx=5, pady=5, expand=True, fill=X)
        #ToolTip(self.spinbox , "Select between 1 an 100 to extract frame each the number selected!Example: Select 30 to extract 1 frame each 30 fps!")

        self.buttonOpen = Button(self.frame, text="Run Extract", padx=5, pady=5, command = self.export_frame)
        self.buttonOpen.pack(padx=5, pady=5, expand=True, fill=X)
          # Open the GUI
        self.window.mainloop()


    def extract_frame(self, file,source,n_frame):

      newDir= self.folder_b+"/"+ file.split(".")[-2]
      print(newDir)
      print(source)
      try:
        os.mkdir(newDir)
        print("Create Folder:"+newDir)
      except OSError:
        print("Folder exists:"+newDir)

      cap = cv2.VideoCapture(source)
      counter_frame = 0
      last_extract=0
      print(n_frame)
      while(True):
            print(counter_frame)
            ret, img= cap.read()
            if(ret == False):
                return
            counter_frame+=1
            print(str(n_frame+last_extract==counter_frame))
            if( n_frame+last_extract==counter_frame):
                cv2.imwrite((newDir+"/"+str(counter_frame)+".png").encode('utf-8').strip(), img)
                last_extract+=n_frame


      cap.release()


    def export_frame(self):

      if self.folder_a is None:
          return
      if self.folder_b is None:
          return
      if self.val_frame is None:
          return
      if self.val_frame==0:
          return

      dir=self.folder_a
      listaArquivo = [f for f in listdir(dir) if isfile(join(dir, f))]
      self.tk.write_log("Init process:"+ str(listaArquivo))
      lista_thread=[]

      start_time = TimeUtils.get_time()
      for r, d, f in os.walk(dir):

        for file in f:
            ext=file.split(".")[-1]
            if  ext =="avi" or "mp4"==ext  :
               t=threading.Thread(target=self.extract_frame,args=(file, os.path.join(r, file),self.val_frame.get()))
               t.start()
               lista_thread.append(t)

      self.tk.append_log("Threads running:"+str(threading.activeCount()))
      for t in (lista_thread):
         t.join()


      end_time = TimeUtils.get_time()
      self.tk.append_log("Finish process:"+str(end_time - start_time))

    def select_folder_source(self):
        self.folder_a=None
        options = {

            'title': 'Auto select all videos from folder(AVI or MP4)',
            'filetypes': (("File MP4", '*.mp4'), ('File AVI', '*.avi'))

        }
        filename = tkFileDialog.askdirectory()
        if(filename != ''):
            self.folder_a = filename
        self.label_a.configure(text=self.folder_a )
        self.window.update_idletasks()

    def select_folder_export(self):
        self.folder_b=None
        options = {

            'title': 'Select folder to export frames(PNG)',
            'filetypes': (('File PNG', '*.png'))

        }
        filename = tkFileDialog.askdirectory()#askopenfilename(**options)
        if(filename != ''):
            self.folder_b = filename
        self.label_b.configure(text=self.folder_b )
        self.window.update_idletasks()
#tela=ExtractFM().run()
