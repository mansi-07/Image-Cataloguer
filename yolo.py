from asyncio.windows_events import NULL
from flask import Flask, render_template, flash, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileAllowed
from wtforms import SubmitField
import os
from image_cataloguer3 import full_model

app=Flask(__name__)
app.config['SECRET_KEY']='568325235'

picFolder= os.path.join('static','pics')
app.config['UPLOAD_FOLDER']=picFolder

class imgForm(FlaskForm):
    p_img = FileField(label='Image Cataloguer',validators=[FileAllowed(['jpeg','png','jpg'])])
    submit = SubmitField(label='Submit')

def save_img(pic):
    picture_name='target.jpg'
    picture_path=os.path.join(app.root_path,'static/input',picture_name)
    pic.save(picture_path)
    return picture_name

@app.route('/',methods=['GET','POST'])
def home_page():
    form = imgForm()
    print(form.validate_on_submit())
    if form.validate_on_submit():
            img_file_name=save_img(form.p_img.data)
            print(form.p_img.type)
            #Since after calling the save_img function the the image is stored in static/input it can be used
            #as input for the model and the model can be instantiated here.and the output of the model can be 
            #put in the static/pics folder.
            tags=full_model()
            # tags=['tag1','tag2','tag3','tag4','tag5']

            pics_temp=os.listdir('C:/Users/Aneesh Kulkarni/web_dev/flask projects/web page for yolo/static/pics')
            pics=[]
            for i in range(len(pics_temp)):
                print(os.listdir('C:/Users/Aneesh Kulkarni/web_dev/flask projects/web page for yolo/static/pics'))
                pics.append(os.path.join(app.config['UPLOAD_FOLDER'],pics_temp[i]))
                print(pics)
            return render_template('show.html',path=app.root_path,tags=tags,user_imgs=pics,length=len(pics))
        # except:
        #     flash('Please put in an image in valid format','error')
        #     return render_template('homepage2.html',form=form)
    if(form.p_img.data is not None):
        print(form.p_img.data)
        flash('Please select an image only','error')
    dir='C:/Users/Aneesh Kulkarni/web_dev/flask projects/web page for yolo/static/pics'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    dir2='C:/Users/Aneesh Kulkarni/web_dev/flask projects/web page for yolo/static/input'
    for f in os.listdir(dir2):
        os.remove(os.path.join(dir2,f))   
    return render_template('homepage2.html',form=form)


@app.route('/about')
def about_page():
    return render_template('about.html')


