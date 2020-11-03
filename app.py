from datetime import datetime
from bson.json_util import dumps
from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, join_room, leave_room
from pymongo.errors import DuplicateKeyError
from flask import send_file
from PIL import Image
from random import randrange
import os
import numpy as np
from db import get_user, save_user, save_room, add_room_members, get_rooms_for_user, get_room, is_room_member,get_room_members, is_room_admin, update_room, remove_room_members, save_message, get_messages
import torch
import torch.nn as nn
from torch.autograd import Variable
from information_hiding import ih_first
from steganalysis import s_first


app = Flask(__name__)
app.secret_key = "sfdjkafnk"
socketio = SocketIO(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)


def padHex(string, length):
    while len(string) < length:
        string = "0" + string
    return string

def padBack(string, length):
    while len(string) < length:
        string += "0"
    return string

#default pixel mapping for AnyImg/ImgAny. Maps row by row, left to right, top to bottom.
def linearMap(n, width, height):
    return (n % width, n // width)




@app.route('/')
def home():
    rooms = []
    try:
        if (current_user.is_authenticated()):
            rooms = get_rooms_for_user(current_user.username)
            return render_template("index.html", rooms=rooms)
    except:
        return render_template("home.html", rooms=rooms)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    message = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password_input = request.form.get('password')
        user = get_user(username)

        if user and user.check_password(password_input):
            login_user(user)
            return redirect(url_for('home'))
        else:
            message = 'Failed to login!'
    return render_template('login.html', message=message)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    message = ''
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            save_user(username, email, password)
            return redirect(url_for('login'))
        except DuplicateKeyError:
            message = "User already exists!"
    return render_template('signup.html', message=message)


@app.route("/logout/")
@login_required
def logout():
    logout_user()
    files = './result1'
    for f in os.listdir(files):
        filepath = os.path.join(files, f)
        os.remove(filepath)
    return redirect(url_for('home'))


@app.route('/create-room/', methods=['GET', 'POST'])
@login_required
def create_room():
    message = ''
    if request.method == 'POST':
        room_name = request.form.get('room_name')
        usernames = [username.strip() for username in request.form.get('members').split(',')]

        if len(room_name) and len(usernames):
            room_id = save_room(room_name, current_user.username)
            if current_user.username in usernames:
                usernames.remove(current_user.username)
            add_room_members(room_id, room_name, usernames, current_user.username)
            return redirect(url_for('view_room', room_id=room_id))
        else:
            message = "Failed to create room"
    return render_template('create_room.html', message=message)


@app.route('/rooms/<room_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_room(room_id):
    room = get_room(room_id)
    if room and is_room_admin(room_id, current_user.username):
        existing_room_members = [member['_id']['username'] for member in get_room_members(room_id)]
        room_members_str = ",".join(existing_room_members)
        message = ''
        if request.method == 'POST':
            room_name = request.form.get('room_name')
            room['name'] = room_name
            update_room(room_id, room_name)

            new_members = [username.strip() for username in request.form.get('members').split(',')]
            members_to_add = list(set(new_members) - set(existing_room_members))
            members_to_remove = list(set(existing_room_members) - set(new_members))
            if len(members_to_add):
                add_room_members(room_id, room_name, members_to_add, current_user.username)
            if len(members_to_remove):
                remove_room_members(room_id, members_to_remove)
            message = 'Room edited successfully'
            room_members_str = ",".join(new_members)
        return render_template('edit_room.html', room=room, room_members_str=room_members_str, message=message)
    else:
        return "Room not found", 404


@app.route('/rooms/<room_id>/')
@login_required
def view_room(room_id):
    rooms = get_rooms_for_user(current_user.username)
    room = get_room(room_id)
    if room and is_room_member(room_id, current_user.username):
        room_members = get_room_members(room_id)
        messages = get_messages(room_id)
        return render_template('view_room.html', username=current_user.username, room=room, room_members=room_members,
                               messages=messages, rooms=rooms)
    else:
        return "Room not found", 404


@app.route('/rooms/<room_id>/messages/')
@login_required
def get_older_messages(room_id):
    room = get_room(room_id)
    if room and is_room_member(room_id, current_user.username):
        page = int(request.args.get('page', 0))
        messages = get_messages(room_id, page)
        return dumps(messages)
    else:
        return "Room not found", 404


@socketio.on('send_message')
def handle_send_message_event(data):
    app.logger.info("{} has sent message to the room {}: {}".format(data['username'],
                                                                    data['room'],
                                                                    data['message']))
    data['created_at'] = datetime.now().strftime("%d %b, %H:%M")
    save_message(data['room'], data['message'], data['username'])
    socketio.emit('receive_message', data, room=data['room'])


@app.route('/save', methods=['GET', 'POST'])
def save():
    return render_template('upload.html')


@app.route('/remove', methods=['POST'])
def remove():
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(APP_ROOT, 'static/')
    listd = os.listdir(target)
    if 'temp.jpg' in listd:
        os.remove(target+'temp.jpg')
    return render_template('upload.html')



@app.route('/upload', methods=['POST'])
def upload():
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(APP_ROOT, 'static/')
    listd = os.listdir(target)
    if 'temp.jpg' in listd:
        os.remove(target+'temp.jpg')
    for file in request.files.getlist("file"):
        filename = file.filename
        filename.split('.')
        destination = '/'.join([target, 'temp.jpg'])
        file.save(destination)
    return render_template('upload.html')


###################################################################################################
@app.route('/savenc', methods=['GET','POST'])
def redEnc():
    return render_template('encryption.html')

@app.route('/encrypt', methods=['POST'])
def encrypt():
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(APP_ROOT, 'static/')
    coverfile=''
    messagefile=''
    for file in request.files.getlist("coverfile"):
        coverfile = file.filename
        coverfile = '/'.join([target, coverfile])
        file.save(coverfile)
    for file in request.files.getlist("messagefile"):
        messagefile = file.filename
        messagefile = '/'.join([target, messagefile])
        file.save(messagefile)
    bits=int(request.form.getlist("bits")[0])
    channel=int(request.form.getlist("channel")[0])

    sec = messagefile
    file = open(sec, "rb").convert("RGBA")
    cov = coverfile
    imgIn = Image.open(cov).convert("RGBA")
    out = '/'.join([target,"stego.png"])
    imgOutPath = out
    b = bits
    bits = abs(int(b))
    style = padHex(bin(abs(int(channel)))[2:], 4)
    R = int(style[3])
    G = int(style[2])
    B = int(style[1])
    A = int(style[0])
    if bits > 8:
        raise Exception("""stego.py does not export more than 8 bits per channel""")
    width, height = imgIn.size
    if (width * height * (bits * (R + G + B + A))) // 8 < os.path.getsize(sec):
        raise Exception("""Can't fit file into image; use more bits or channels.""")
    key = chr(0)
    binString = ""
    b = file.read(1)
    while b:
        b = int.from_bytes(b, byteorder="big", signed=False) ^ ord(key[0])
        key = key[1:] + key[0]
        binString = binString + padHex(bin(b)[2:], 8)
        b = file.read(1)

    binString = binString + "0" * 256
    lsbList = []
    while len(binString) > 0:
        pix = [0, 0, 0, 0]
        if R and binString[:bits] != "":
            pix[0] = int(binString[:bits], 2)
            binString = binString[bits:]
        if G and binString[:bits] != "":
            pix[1] = int(binString[:bits], 2)
            binString = binString[bits:]
        if B and binString[:bits] != "":
            pix[2] = int(binString[:bits], 2)
            binString = binString[bits:]
        if A and binString[:bits] != "":
            pix[3] = int(binString[:bits], 2)
            binString = binString[bits:]
        lsbList.append(pix)
    while len(lsbList) < width * height:
        lsbList.append([randrange(2 ** bits), randrange(2 ** bits), randrange(2 ** bits), randrange(2 ** bits)])
    for n in range(width * height):
        pixXY = linearMap(n, width, height)
        r, g, b, a = imgIn.getpixel(pixXY)
        for i in range(bits):
            r -= 2 ** i * int(padHex(bin(r)[2:], 8)[::-1][i:i + 1]) * R
            g -= 2 ** i * int(padHex(bin(g)[2:], 8)[::-1][i:i + 1]) * G
            b -= 2 ** i * int(padHex(bin(b)[2:], 8)[::-1][i:i + 1]) * B
            a -= 2 ** i * int(padHex(bin(a)[2:], 8)[::-1][i:i + 1]) * A
        r += lsbList[n][0] * R
        g += lsbList[n][1] * G
        b += lsbList[n][2] * B
        a += lsbList[n][3] * A
        imgIn.putpixel(pixXY, (r, g, b, a))

    imgIn.save(imgOutPath, "PNG")
    print("finish!")
    os.remove(messagefile)
    os.remove(coverfile)
    return send_file(imgOutPath, as_attachment=True)


@app.route('/savdec', methods=['GET','POST'])
def redDec():
    return render_template('decryption.html')

@app.route('/decrypt', methods=['POST'])
def decrypt():
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(APP_ROOT, 'static/')
    stegofile=''
    for file in request.files.getlist("stegoimage"):
        stegofile = file.filename
        stegofile = '/'.join([target, stegofile])
        file.save(stegofile)

    bits = int(request.form.getlist("bits")[0])
    channel = int(request.form.getlist("channel")[0])

    # in_file = input("Enter the file:")
    imgIn = Image.open(stegofile).convert("RGBA")
    out_file = '/'.join([target,"smessage.png"])
    file = open(out_file, "wb")
    # b = input("Enter the bits:")
    bits = abs(bits)
    # channel = input("Enter channels:")
    style = padHex(bin(abs(int(channel)))[2:], 4)
    R = int(style[3])
    G = int(style[2])
    B = int(style[1])
    A = int(style[0])
    if bits > 8:
        raise Exception("""stego.py does not support more than 8 bits per channel""")
    key = chr(0)
    # key_choice = int(input("Want to enter key? 1 for yes"))
    # if key_choice == 1:
    #     key = input("Enter key:")
    # else:
    #     print("Key will be:", key)

        # first extract out all lsb data
    binString = ""
    width, height = imgIn.size
    for n in range(width * height):
        pixXY = linearMap(n, width, height)
        r, g, b, a = imgIn.getpixel(pixXY)
        dr = 0
        dg = 0
        db = 0
        da = 0
        for i in range(bits):
            dr += 2 ** i * int(padHex(bin(r)[2:], 8)[::-1][i:i + 1])
            dg += 2 ** i * int(padHex(bin(g)[2:], 8)[::-1][i:i + 1])
            db += 2 ** i * int(padHex(bin(b)[2:], 8)[::-1][i:i + 1])
            da += 2 ** i * int(padHex(bin(a)[2:], 8)[::-1][i:i + 1])
        binString += (padHex(bin(dr)[2:], bits) if R else "") + (padHex(bin(dg)[2:], bits) if G else "") + (
            padHex(bin(db)[2:], bits) if B else "") + (padHex(bin(da)[2:], bits) if A else "")
    # now, apply xor and write string back into file
    while binString[:256] != "0" * 256:
        b = int(binString[:8], 2) ^ ord(key[0])
        key = key[1:] + key[0]
        file.write(b.to_bytes(1, byteorder="big"))
        binString = binString[8:]
    file.close()
    print("finish!")
    os.remove(stegofile)

    return send_file(out_file, as_attachment=True)



@socketio.on('join_room')
def handle_join_room_event(data):
    app.logger.info("{} has joined the room {}".format(data['username'], data['room']))
    join_room(data['room'])
    socketio.emit('join_room_announcement', data, room=data['room'])


@socketio.on('leave_room')
def handle_leave_room_event(data):
    app.logger.info("{} has left the room {}".format(data['username'], data['room']))
    leave_room(data['room'])
    socketio.emit('leave_room_announcement', data, room=data['room'])


@login_manager.user_loader
def load_user(username):
    return get_user(username)


@app.route('/tnst', methods=['GET','POST'])
def tnst():
    return render_template('nst.html')





@app.route('/nst', methods=['POST'])
def nst():
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(APP_ROOT, 'static/')
    content = ''
    style = ''
    for file in request.files.getlist("content"):
        content = file.filename
        content = '/'.join([target, content])
        file.save(content)
    for file in request.files.getlist("style"):
        style = file.filename
        style = '/'.join([target, style])
        file.save(style)

    out_file = '/'.join([target, "output.png"])

    class GramMatrix(nn.Module):
        def forward(self, y):
            (b, ch, h, w) = y.size()
            features = y.view(b, ch, w * h)
            features_t = features.transpose(1, 2)
            gram = features.bmm(features_t) / (ch * h * w)
            return gram

    # proposed Inspiration(CoMatch) Layer
    class Inspiration(nn.Module):
        """ Inspiration Layer (from MSG-Net paper)
        tuning the featuremap with target Gram Matrix
        ref https://arxiv.org/abs/1703.06953
        """

        def __init__(self, C, B=1):
            super(Inspiration, self).__init__()
            # B is equal to 1 or input mini_batch
            self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)
            # non-parameter buffer
            self.G = Variable(torch.Tensor(B, C, C), requires_grad=True)
            self.C = C
            self.reset_parameters()

        def reset_parameters(self):
            self.weight.data.uniform_(0.0, 0.02)

        def setTarget(self, target):
            self.G = target

        def forward(self, X):
            # input X is a 3D feature map
            self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
            return torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
                             X.view(X.size(0), X.size(1), -1)).view_as(X)

        def __repr__(self):
            return self.__class__.__name__ + '(' \
                   + 'N x ' + str(self.C) + ')'

    # some basic layers, with reflectance padding
    class ConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride):
            super(ConvLayer, self).__init__()
            reflection_padding = int(np.floor(kernel_size / 2))
            self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        def forward(self, x):
            out = self.reflection_pad(x)
            out = self.conv2d(out)
            return out

    class UpsampleConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
            super(UpsampleConvLayer, self).__init__()
            self.upsample = upsample
            if upsample:
                self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
            self.reflection_padding = int(np.floor(kernel_size / 2))
            if self.reflection_padding != 0:
                self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        def forward(self, x):
            if self.upsample:
                x = self.upsample_layer(x)
            if self.reflection_padding != 0:
                x = self.reflection_pad(x)
            out = self.conv2d(x)
            return out

    class Bottleneck(nn.Module):
        """ Pre-activation residual block
        Identity Mapping in Deep Residual Networks
        ref https://arxiv.org/abs/1603.05027
        """

        def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
            super(Bottleneck, self).__init__()
            self.expansion = 4
            self.downsample = downsample
            if self.downsample is not None:
                self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion,
                                                kernel_size=1, stride=stride)
            conv_block = []
            conv_block += [norm_layer(inplanes),
                           nn.ReLU(inplace=True),
                           nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
            conv_block += [norm_layer(planes),
                           nn.ReLU(inplace=True),
                           ConvLayer(planes, planes, kernel_size=3, stride=stride)]
            conv_block += [norm_layer(planes),
                           nn.ReLU(inplace=True),
                           nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
            self.conv_block = nn.Sequential(*conv_block)

        def forward(self, x):
            if self.downsample is not None:
                residual = self.residual_layer(x)
            else:
                residual = x
            return residual + self.conv_block(x)

    class UpBottleneck(nn.Module):
        """ Up-sample residual block (from MSG-Net paper)
        Enables passing identity all the way through the generator
        ref https://arxiv.org/abs/1703.06953
        """

        def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
            super(UpBottleneck, self).__init__()
            self.expansion = 4
            self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
                                                    kernel_size=1, stride=1, upsample=stride)
            conv_block = []
            conv_block += [norm_layer(inplanes),
                           nn.ReLU(inplace=True),
                           nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)]
            conv_block += [norm_layer(planes),
                           nn.ReLU(inplace=True),
                           UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride)]
            conv_block += [norm_layer(planes),
                           nn.ReLU(inplace=True),
                           nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)]
            self.conv_block = nn.Sequential(*conv_block)

        def forward(self, x):
            return self.residual_layer(x) + self.conv_block(x)

    # the MSG-Net
    class Net(nn.Module):
        def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=6, gpu_ids=[]):
            super(Net, self).__init__()
            self.gpu_ids = gpu_ids
            self.gram = GramMatrix()

            block = Bottleneck
            upblock = UpBottleneck
            expansion = 4

            model1 = []
            model1 += [ConvLayer(input_nc, 64, kernel_size=7, stride=1),
                       norm_layer(64),
                       nn.ReLU(inplace=True),
                       block(64, 32, 2, 1, norm_layer),
                       block(32 * expansion, ngf, 2, 1, norm_layer)]
            self.model1 = nn.Sequential(*model1)

            model = []
            self.ins = Inspiration(ngf * expansion)
            model += [self.model1]
            model += [self.ins]

            for i in range(n_blocks):
                model += [block(ngf * expansion, ngf, 1, None, norm_layer)]

            model += [upblock(ngf * expansion, 32, 2, norm_layer),
                      upblock(32 * expansion, 16, 2, norm_layer),
                      norm_layer(16 * expansion),
                      nn.ReLU(inplace=True),
                      ConvLayer(16 * expansion, output_nc, kernel_size=7, stride=1)]

            self.model = nn.Sequential(*model)

        def setTarget(self, Xs):
            F = self.model1(Xs)
            G = self.gram(F)
            self.ins.setTarget(G)

        def forward(self, input):
            return self.model(input)

    ###############################################################################################################################
    def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
        img = Image.open(filename).convert('RGB')
        if size is not None:
            if keep_asp:
                size2 = int(size * 1.0 / img.size[0] * img.size[1])
                img = img.resize((size, size2), Image.ANTIALIAS)
            else:
                img = img.resize((size, size), Image.ANTIALIAS)

        elif scale is not None:
            img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
        img = np.array(img).transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

    def tensor_save_rgbimage(tensor, filename, cuda=False):
        if cuda:
            img = tensor.clone().cpu().clamp(0, 255).numpy()
        else:
            img = tensor.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype('uint8')
        img = Image.fromarray(img)
        img.save(filename)

    def tensor_save_bgrimage(tensor, filename, cuda=False):
        (b, g, r) = torch.chunk(tensor, 3)
        tensor = torch.cat((r, g, b))
        tensor_save_rgbimage(tensor, filename, cuda)

    def preprocess_batch(batch):
        batch = batch.transpose(0, 1)
        (r, g, b) = torch.chunk(batch, 3)
        batch = torch.cat((b, g, r))
        batch = batch.transpose(0, 1)
        return batch

    content_image = tensor_load_rgbimage(content, size=512, keep_asp=True).unsqueeze(0)
    style = tensor_load_rgbimage(style, size=512).unsqueeze(0)
    style = preprocess_batch(style)

    model_dict = torch.load('21styles.model')
    model_dict_clone = model_dict.copy()  # We can't mutate while iterating

    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]

    ### Next cell

    style_model = Net(ngf=128)
    style_model.load_state_dict(model_dict, False)

    # style_model = Net(ngf=128)
    # style_model.load_state_dict(torch.load('21styles.model'), False)

    style_v = Variable(style)
    content_image = Variable(preprocess_batch(content_image))
    style_model.setTarget(style_v)
    output = style_model(content_image)
    tensor_save_bgrimage(output.data[0], out_file, False)
    imgin = Image.open(out_file).convert("RGBA")
    imgin.save(out_file,"PNG")
    os.remove(content)
    os.remove(style)
    return send_file(out_file, as_attachment=True)


@app.route('/deepenc', methods=['GET','POST'])
def dEnc():
    return render_template('dencryption.html')

@app.route('/dencrypt', methods=['POST'])
def dencrypt():
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(APP_ROOT, 'static/')
    coverfile=''
    messagefile=''
    for file in request.files.getlist("coverfile"):
        coverfile = file.filename
        coverfile = '/'.join([target, coverfile])
        file.save(coverfile)
    for file in request.files.getlist("messagefile"):
        messagefile = file.filename
        messagefile = '/'.join([target, messagefile])
        file.save(messagefile)

    ih_first(messagefile,coverfile)
    os.remove(messagefile)
    os.remove(coverfile)
    return send_file('./result1/output.png', as_attachment=True)


@app.route('/deepdec', methods=['GET','POST'])
def dDec():
    return render_template('ddecryption.html')

@app.route('/ddecrypt', methods=['POST'])
def ddecrypt():
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(APP_ROOT, 'static/')
    stegofile=''
    for file in request.files.getlist("stegoimage"):
        stegofile = file.filename
        stegofile = '/'.join([target, stegofile])
        file.save(stegofile)

    s_first(stegofile)
    os.remove(stegofile)
    return send_file('./result1/reveal_secret.png', as_attachment=True)




if __name__ == '__main__':
    socketio.run(app, debug=True)