import subprocess
import pdb
import argparse
import cv2 
import numpy as np
import os
from os import remove
import time
class deteccion():
  def __init__(self, nombreClase, idClase, bbox, score):
    self.nombreClase = nombreClase
    self.idClase = idClase
    self.bbox = bbox
    self.score = score
    

def devuelveNombreDeClases(path):
  datos = []
  with open(path) as fname:
	  lineas = fname.readlines()
	  for linea in lineas:
		  datos.append(linea.strip('\n'))
  return datos

def tratamientoResultado(direccionImagenes, nombreImagen ,model,thread):

  clases = devuelveNombreDeClases("./coco.names")
  result = subprocess.run(["./test_jpeg_yolov4", model, direccionImagenes, str(thread)], stderr=subprocess.PIPE, text=True)
  #pdb.set_trace()
  files_names_borrar = os.listdir("./")
  filenameABorrar = "0_" + nombreImagen.split(".")[0] + "_result." +nombreImagen.split(".")[1]
  if filenameABorrar in files_names_borrar:
      remove(filenameABorrar)
  #pdb.set_trace()
  ListaResult = result.stderr.split("RESULT:")[1:]
  resutadosCompletos = []
  for prediccion in ListaResult:
      listaPrediccion = prediccion.split("\t")
      nombreClase = clases[int(listaPrediccion[0])]
      #pdb.set_trace()
      bbox = np.array([np.float(listaPrediccion[1]), np.float(listaPrediccion[2]), np.float(listaPrediccion[3]), np.float(listaPrediccion[4])], dtype=float)
      score = (listaPrediccion[5].split("\n"))[0]
      resutadosCompletos.append(deteccion(nombreClase,int(listaPrediccion[0]),bbox,score))
  return resutadosCompletos

def draw_boxes(image, listaDeteccion):
  image_h, image_w, _ = image.shape
  diccioanrioDeteccion = {}
  for deteccion in listaDeteccion:
      xmin = int(deteccion.bbox[0])
      ymin = int(deteccion.bbox[1])
      xmax = int(deteccion.bbox[2])
      ymax = int(deteccion.bbox[3])
      # pdb.set_trace()
      color = list(np.random.choice(range(256), size=3))
      if deteccion.idClase in diccioanrioDeteccion:
        color = diccioanrioDeteccion[deteccion.idClase]
      else:
        diccioanrioDeteccion[deteccion.idClase] = color
      
      cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (int(color[0]), int(color[1]),int(color[2])), 3)
      cv2.putText(image, 
                  deteccion.nombreClase +' '+ str(deteccion.score),
                  (xmin + 20, ymin - 13), 
                  cv2.FONT_HERSHEY_SIMPLEX, 
                  1e-3 * image_h, 
                  (int(color[0]), int(color[1]),int(color[2])), int(1e-3 * image_h)) 
  return image  

def app(dir_file,model,thread, eval = False):
  #se hace una transformación para coger el nombre de la imagen si llega una dirección
  if "/" in dir_file:
    dirImagen = dir_file.split("/")
    nombreImagen = dirImagen[len(dirImagen)-1]
  else: 
    nombreImagen = dir_file
  dir_file.split("/")
  # try:

  resultados = tratamientoResultado(dir_file, nombreImagen,model,thread)
  if not eval:
    image_path = dir_file
    img = cv2.imread(image_path)
    imagenBBOX = draw_boxes(img,resultados)
    nombreImagenEscritura = "resultado_" + nombreImagen
    cv2.imwrite(nombreImagenEscritura, imagenBBOX)
  return resultados
  # except:
  #   print("La imagen pasada no existe o es un directorio erroneo")



def main():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--image_dir', type=str, default='cepillo.jpg', help='Path to folder of images. Default is images')  
  ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',     type=str, default='yolov4_leaky_spp_m', help='Path of xmodel. Default is model_dir/customcnn.xmodel')

  args = ap.parse_args()  
  
  app(args.image_dir,args.model,args.threads)
  # tratamientoResultado(args.image_dir,args.threads,args.model)

if __name__ == '__main__':
  main()
