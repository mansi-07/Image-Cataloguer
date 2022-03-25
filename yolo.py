from __future__ import print_function
from asyncio.windows_events import NULL
from flask import Flask, render_template, flash, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileAllowed
from wtforms import SubmitField
import os
from image_cataloguer3 import full_model
from color import colors
app=Flask(__name__)
app.config['SECRET_KEY']='568325235'

picFolder= os.path.join('static','pics')
app.config['UPLOAD_FOLDER']=picFolder

dict_i={}
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
    # print(form.validate_on_submit())
    if form.validate_on_submit():
        try:
            img_file_name=save_img(form.p_img.data)
            s_tags=full_model()
            tags=list(s_tags)
            c_tags=colors(r'C:\Users\Aneesh Kulkarni\web_dev\flask projects\web page for yolo\static\input\target.jpg')
            if(len(tags)<5):
                c_tags=c_tags[0:5-len(tags)]
            pics_temp=os.listdir('C:/Users/Aneesh Kulkarni/web_dev/flask projects/web page for yolo/static/pics')
            pics=[]
            for i in range(len(pics_temp)):
                # print(os.listdir('C:/Users/Aneesh Kulkarni/web_dev/flask projects/web page for yolo/static/pics'))
                pics.append(os.path.join(app.config['UPLOAD_FOLDER'],pics_temp[i]))
            # print(pics)
            # print(type(pics[0]))
            #now running colors model all images in the pics folder
            for pic in pics:
                dict_i[pic]=colors(os.path.join(r'C:\Users\Aneesh Kulkarni\web_dev\flask projects\web page for yolo',pic))
            ###
            # print(dict_i)
            return render_template('show.html',path=app.root_path,tags=tags,c_tags=c_tags,user_imgs=pics,length=len(pics))
        except:
            flash('Please put in an image in valid format','error')
            return render_template('homepage2.html',form=form)


    if(form.p_img.data is not None):
        # print(form.p_img.data)
        flash('Please select an image only','error')
    dir='C:/Users/Aneesh Kulkarni/web_dev/flask projects/web page for yolo/static/pics'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    dir2='C:/Users/Aneesh Kulkarni/web_dev/flask projects/web page for yolo/static/input'
    for f in os.listdir(dir2):
        os.remove(os.path.join(dir2,f))   
    return render_template('homepage2.html',form=form)


@app.route('/display/<tag>')
def display(tag):
    try:
        arr=os.listdir(f'C:/Users/Aneesh Kulkarni/web_dev/flask projects/web page for yolo/static/cataloguer/{tag}')
        imgs=[]
        for i in range(25):
            imgs.append(tag+'/'+arr[i])
        res=1
        return render_template('display.html',tag=tag,imgs=imgs,res=res)
    except:
        res=0
        return render_template('display.html',res=res)


@app.route('/display/color/<tag>')
def c_display(tag):
    pics_temp=os.listdir('C:/Users/Aneesh Kulkarni/web_dev/flask projects/web page for yolo/static/pics')
    pics=[]
    for i in range(len(pics_temp)):
        pics.append(os.path.join(app.config['UPLOAD_FOLDER'],pics_temp[i]))
    # print(pics)
    # print(dict_i)
    arr=[]
    for pic in pics:
        if tag in dict_i[pic]:
            s=pic[12:]
            # print(s)
            # print(type(s))
            arr.append(s)
    # print(arr)
    return render_template('c_display.html',imgs=arr,tag=tag)
        

@app.route('/about')
def about_page():
    return render_template('about.html')


