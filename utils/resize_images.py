from PIL import Image
import glob

#resize delle immagini per il vit (224x224x3)
baseheight = 1200
width = 1200
img_path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\Images\\*'
newimg_path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\ResizedImages\\'
images = glob.glob(img_path)

for path in images:
    img = Image.open(path)
    img = img.resize((width, baseheight))
    img.save(newimg_path + path[19:])