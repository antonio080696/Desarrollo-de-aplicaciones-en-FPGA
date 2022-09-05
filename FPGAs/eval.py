import app 
import pdb
import argparse
import json 

bbox_output = 'bbox_out.json'

bboxpred = []
def app_eval(eval, imagesDir, model, threads):
	with open(eval) as json_file:
		data = json.load(json_file)
		cnt = 0
		for i in data['images']:
			cnt=cnt+1
			print("processing: ", cnt, "of 5000 images")
			img_file_name = i['file_name']
			metadata = {'id': i['id'], 'height': i['height'], 'width': i['width']}
			img_path=imagesDir + img_file_name
			lista_Deteccion =app.app(img_path,model,threads, True)
			bbox = (deteccion.bbox[0],deteccion.bbox[1],(deteccion.bbox[2]-deteccion.bbox[0]),(deteccion.bbox[3]-deteccion.bbox[1]))
			#aplicamos el redondeo
			bbox = [np.round(b * 100) / 100 for b in bbox]
			bbox = [b.item() for b in bbox]
			score = float(deteccion.score)
			score = np.round(score * 10000) / 10000
			score = score.item()
			
			id_clase = convert_coco_category(deteccion.idClase)
			for deteccion in lista_Deteccion:
				bboxpred.append({'image_id': int(i['id']), 'category_id': id_clase, 'bbox': bbox, 'score': score})
				if len(deteccion.bbox) == 1:
					single_bboxes_id.append(int(i['id']))
					bboxpred.append({'image_id': int(i['id']), 'category_id': int(0), 'bbox': bbox, 'score': 0})
			if cnt == 1 :
				del(metadata)
				break

	with open(bbox_output, 'w') as json_file:
		json.dump(bboxpred, json_file)
def convert_coco_category(category_id):
    '''
    convert continuous coco class id (0~79) to discontinuous coco category id
    '''
    if category_id >= 0 and category_id <= 10:
        category_id = category_id + 1
    elif category_id >= 11 and category_id <= 23:
        category_id = category_id + 2
    elif category_id >= 24 and category_id <= 25:
        category_id = category_id + 3
    elif category_id >= 26 and category_id <= 39:
        category_id = category_id + 5
    elif category_id >= 40 and category_id <= 59:
        category_id = category_id + 6
    elif category_id == 60:
        category_id = category_id + 7
    elif category_id == 61:
        category_id = category_id + 9
    elif category_id >= 62 and category_id <= 72:
        category_id = category_id + 10
    elif category_id >= 73 and category_id <= 79:
        category_id = category_id + 11
    else:
        raise ValueError('Invalid category id')
    return category_id

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-e', '--eval',   type=str, default='instances_val2017.json', help='archivo de evaluacion .json')
	ap.add_argument('-t', '--threads',   type=int, default=1, help='Numero de hilos. por defecto es 1')
	ap.add_argument('-m', '--model',     type=str, default='yolov4_leaky_spp_m', help='Path del .xmodel a usar. Por defecto es yolov4_leaky_spp_m')
	ap.add_argument('-d','--data_dir', type=str, default='./val2017/', required=False, help='path a la carpeta que contiene las imagenes de evaluacion')
	args = ap.parse_args()  
	app_eval(args.eval,args.data_dir,args.model,args.threads)
if __name__ == '__main__':
  main()