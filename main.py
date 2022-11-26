from kivy.lang import Builder
from kivymd.app  import MDApp
from kivy.config import Config
from kivymd.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
import os
import numpy as np

from kivy.clock import Clock

Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '600') # mudar para 1024
Config.write()

class Front(FloatLayout):
    ...


class Teclado(FloatLayout):
    ...


class KivyCamera(Image):
    def __init__(self, capture, fps, cascade, model, names, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)
        self.cascade = cascade
        self.model = model
        self.names = names

    def update(self, dt):
        ...
        #faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")  # added

        ret, frame = self.capture.read()
        if ret:

            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # added
            faces = self.cascade.detectMultiScale(frameGray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = frameGray[y:y + h, x:x + w]
                # Try to recognize the face
                face_resize = cv2.resize(face, (130, 100))
                prediction = self.model.predict(face_resize)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                if prediction[1] < 500:
                    #print(self.names)
	                cv2.putText(frame, '% s - %.0f' % (self.names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                else:
                    cv2.putText(frame, 'not recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture



class HomemadeWebcam(MDApp):
    KV_FILES = ['app/teclado.kv', 'app/main.kv', 'app/manager.kv']
    DEBUG = True
    def __init__(self):
        super().__init__()
        self.kv_dir = 'app'
        for file in os.listdir(self.kv_dir):
            if file != 'manager.kv':
                Builder.load_file(os.path.join(self.kv_dir, file))

        self.manager = Builder.load_file('app/manager.kv')
        self.main = self.manager.ids['main']
        self.teclado = self.manager.ids['teclado']
        
        datasets = 'datasets'
        self.images = []
        self.labels = []
        self.names = {}
        self.id = 0
        for (subdirs, dirs, files) in os.walk(datasets):
            for subdir in dirs:
                self.names[self.id] = subdir
                subjectpath = os.path.join(datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    label = self.id
                    self.images.append(cv2.imread(path, 0))
                    self.labels.append(int(label))
                self.id += 1
        (self.images, self.labels) = [np.array(lis) for lis in [self.images, self.labels]]

        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.train(self.images, self.labels)
        self.cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.capture = cv2.VideoCapture(0)
        self.cam = KivyCamera(self.capture, 60, self.cascade, self.model, self.names, size=(700,600))
        self.main.add_widget(self.cam)
        
        

    def build(self):
        return self.manager

    def on_stop(self):
        self.capture.release()
    
    def muda_tela(self):
        if self.root.current == 'main':
            self.root.current = 'teclado'
            self.main.remove_widget(self.cam)
            self.teclado.add_widget(self.cam)
        elif self.root.current == 'teclado':
            self.root.current = 'main'
            self.teclado.remove_widget(self.cam)
            self.main.add_widget(self.cam)

HomemadeWebcam().run()
