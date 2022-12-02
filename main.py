from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.config import Config
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
import os
import numpy as np
from kivymd.uix.label import MDLabel
from kivy.uix.widget import Widget

from kivy.clock import Clock

Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '600')  # mudar para 1024
Config.write()

colors = {
    'Teal': {
        '200': '#212121',
        '500': '#212121',
        '700': '#212121',
    },
    'Red': {
        '200': '#C25554',
        '500': '#C25554',
        '700': '#C25554',
        'A700': '#C25554',
    },
    'Light': {
        'StatusBar': '#E0E0E0',
        'AppBar': '#AAAAAA',
        'Background': '#AAAAAA',  # Fundo da tela
        'CardsDialogs': '#FFFFFF',
        'FlatButtonDown': '#CCCCCC',
    }
}

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
        # faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")  # added

        ret, frame = self.capture.read()
        #frame = cv2.resize(frame, (780, 640))
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
                    cv2.putText(frame, '% s - %.0f' % (self.names[prediction[0]], prediction[1]), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                else:
                    cv2.putText(frame, 'not recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture

            self.texture = image_texture

class KeyBoardListener(Widget):
    def __init__(self, label_container, label, motivo, de_cima, **kwargs):
        super().__init__(**kwargs)
        self._keyboard = Window.request_keyboard(
            self._keyboard_close, self, 'text')
        if self._keyboard.widget:
            # If it exists, this widget is a VKeyboard object which you can use
            # to change the keyboard layout.
            pass
        self._keyboard.bind(on_key_up=self.on_key_up)
        self.label_container = label_container
        self.display_label = label
        self.motivo = motivo
        self.de_cima = de_cima

    def _keyboard_close(self, *args):
        """ The active keyboard is being closed. """
        if self._keyboard:
            self._keyboard.unbind(on_key_up=self.on_key_up)
            self._keyboard = None
            self.label_container.size_hint = [0, 0]
            self.label_container.children = []
            self.display_label.text = ""
            self.display_label = None


    def on_key_up(self, keyboard, keycode, *args):
        """ The callback function that catches keyboard events. """
        # system keyboard keycode: (122, 'z')
        # dock keyboard keycode: 'z'
        if isinstance(keycode, tuple):
            keycode = keycode[1]

        if keycode == 'enter':
            texto = self.display_label.text[6:]
            if self.motivo == "cadastro":
                self.de_cima.cadastro(texto)
            elif self.motivo == "remover":
                self.de_cima.remover(texto)
            self._keyboard.release()
            return

        elif keycode == 'escape':
            self._keyboard.release()
            return

        elif keycode == 'backspace':
            if len(self.display_label.text) > 6:
                self.display_label.text = self.display_label.text[:-1]
            return

        else:
            self.display_label.text += keycode



class HomemadeWebcam(MDApp):
    DEBUG = True

    def __init__(self):
        super().__init__()
        self.kv_dir = 'app'

        # Configuração de widgets
        self.main = Builder.load_file('app/main.kv')
        self.cam_container = self.main.ids['camera']
        self.display_label_container = self.main.ids['display']
        self.display_label = None

        # Configuração da senha numérica
        self.senha_container = self.main.ids['senha']
        self.digitado = ""
        self.senha = "0"

        # Configuração de teclado
        self._keyboard = None

        # Configuração do reconhecimento
        self.datasets = 'datasets'
        self.model, self.names = self.treina()
        self.haar = "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(self.haar)

        # Configuração da câmera
        self.capture = cv2.VideoCapture(0)
        self.cam = KivyCamera(self.capture, 60, self.cascade, self.model, self.names, size=(700, 600))
        self.cam_container.add_widget(self.cam)

    def build(self):
        # Setup theme
        self.theme_cls.colors = colors
        self.theme_cls.primary_palette = 'Teal'
        self.theme_cls.accent_palette = 'Red'
        return self.main

    def on_stop(self):
        self.capture.release()

    def matriz(self, valor):
        if len(self.digitado) <= 10:
            self.senha_container.text += "*"
            self.digitado += str(valor)

    def abre(self):
        if self.checa_senha():
            print('Acesso liberado!')
    def checa_senha(self):
        if self.digitado == self.senha:
            print('Senha correta!')
            self.digitado = ""
            self.senha_container.text = ""
            return True
        return False

    def delete(self):
        if len(self.digitado) > 0:
            self.senha_container.text = self.senha_container.text[:-1]
            self.digitado = self.digitado[:-1]

    def teclado(self, motivo):
        if not self.checa_senha():
            print('Só é possível cadastrar novos usuários após colocar a senha numérica')
            return

        self.display_label_container.size_hint = [1, .1]
        self.display_label = MDLabel(text='Nome: ', pos_hint={'center_x': .5, 'center_y': .5})
        self.display_label_container.add_widget(self.display_label)
        KeyBoardListener(self.display_label_container, self.display_label, motivo, self)



    def cadastro(self, texto):
        sub_data = texto
        caminho = os.path.join(self.datasets, sub_data)
        if not os.path.isdir(caminho):
            os.makedirs(caminho)

        (width, height) = (130, 100)
        face_cascade = cv2.CascadeClassifier(self.haar)
        count = 1
        while count < 200:
            (_, im) = self.capture.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite('% s/% s.png' % (caminho, count), face_resize)
            count += 1

            cv2.imshow('Criando database do usuario', im)
            key = cv2.waitKey(10)
            if key == 27:
                cv2.destroyWindow('Criando database do usuario')
                break
        cv2.destroyWindow('Criando database do usuario')
        self.model, self.names = self.treina()

    def remover(self, texto):
        caminho = os.path.join(self.datasets, texto)
        if os.path.exists(caminho):
            for file in os.listdir(caminho):
                os.remove(os.path.join(caminho, file))
            os.rmdir(caminho)
            self.treina()
            return
        print('O usuário informado não existe')

    def treina(self):
        images = []
        labels = []
        names = {}
        idx = 0
        for (subdirs, dirs, files) in os.walk(self.datasets):
            for subdir in dirs:
                names[idx] = subdir
                subjectpath = os.path.join(self.datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    label = idx
                    images.append(cv2.imread(path, 0))
                    labels.append(int(label))
                idx += 1
        (images, labels) = [np.array(lis) for lis in [images, labels]]

        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images, labels)
        return model, names




HomemadeWebcam().run()
