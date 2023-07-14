from PIL import Image, ImageDraw, ImageFont

class ImageUtilityPIL:

    def __init__(self,image_route):
        self.image_route =image_route
    
    def mostrarImagen(self,image_name):
        with Image.open(self.image_route + image_name, 'r') as img:
            img.show()