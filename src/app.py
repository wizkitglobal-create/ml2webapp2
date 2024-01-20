import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import base64
import io
from roboflow import Roboflow
from datetime import datetime, date, timedelta
import boto3
from botocore.exceptions import NoCredentialsError
from PIL import Image, ImageDraw, ImageFont
from exif import Image as ExifImage
import numpy as np
import pandas as pd
from boto3.dynamodb.conditions import Key
import urllib.parse
import os

aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
region_name = 'ap-southeast-1'

dynamodb = boto3.resource('dynamodb',
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key,
                          region_name=region_name)
s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name=region_name)

# Initialize Roboflow model
rf = Roboflow(api_key="4mk8lPHkiCrauugzg3jb")
project = rf.workspace().project("ml2-wcn-ukm")
model = project.version(16).model

aggregate_table = dynamodb.Table('AggregateData')
records_table = dynamodb.Table('Records')

app= dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
server=app.server

from PIL import ImageOps

def apply_exif_orientation(image):
    try:
        image_exif = image._getexif()
        if image_exif is not None:
            orientation_key = 274  # EXIF orientation tag
            if orientation_key in image_exif:
                orientation = image_exif[orientation_key]
                rotated_image = ImageOps.exif_transpose(image)
                return rotated_image
    except Exception as e:
        print(f"Error applying EXIF orientation: {e}")

    return image
def convert_image_to_np_array(contents, resize_factor=0.60):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))

    # Apply EXIF orientation correction
    image = apply_exif_orientation(image)

    # Resize image
    image = resize_image(image, resize_factor)

    return np.array(image)

def resize_image(image, resize_factor=0.60):
    new_size = (int(image.size[0] * resize_factor), int(image.size[1] * resize_factor))
    return image.resize(new_size, Image.ANTIALIAS)
def convert_image_to_np_array(contents, resize_factor=0.60):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))

    # Resize image
    image = resize_image(image, resize_factor)

    return np.array(image)


def decimal_coords(coords, ref):
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -decimal_degrees
    return decimal_degrees

def get_image_coordinates(image_bytes):
    image_bytes.seek(0)  # Make sure we're at the start of the BytesIO object
    img = ExifImage(image_bytes)

    if img.has_exif:
        try:
            lat = decimal_coords(img.gps_latitude, img.gps_latitude_ref)
            lon = decimal_coords(img.gps_longitude, img.gps_longitude_ref)
            return lat, lon
        except AttributeError:
            print('No Coordinates Found.')
            return None, None
    else:
        print('The Image has no EXIF information')
        return None, None

def upload_to_s3(image_bytes_io, bucket, s3_file):
    s3 = boto3.client('s3', region_name='ap-southeast-1')

    try:
        # Use 'upload_fileobj' method to upload an in-memory file-like object
        s3.client.upload_fileobj(image_bytes_io, bucket, s3_file)
        print("Upload Successful")
        return True

    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print("An unexpected error occurred:", e)  # added this line to catch all other exceptions
        return False



def parse_contents(contents, prediction, resize_factor=0.60):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))

    # Apply EXIF orientation correction
    image = apply_exif_orientation(image)

    # Resize image for Roboflow inference
    original_size = image.size
    new_size = (int(original_size[0] * resize_factor), int(original_size[1] * resize_factor))
    resized_image = image.resize(new_size, Image.ANTIALIAS)

    # Prepare to draw on the resized image
    draw = ImageDraw.Draw(resized_image)
    font = ImageFont.load_default()

    for obj in prediction['predictions']:
        # Adjust bounding box coordinates for resized image
        x1 = obj['x'] - obj['width'] / 2
        y1 = obj['y'] - obj['height'] / 2
        x2 = obj['x'] + obj['width'] / 2
        y2 = obj['y'] + obj['height'] / 2

        label = obj['class']
        text_width, text_height = draw.textsize(label, font=font)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)

    # Convert resized image to base64 for displaying
    buffered = io.BytesIO()
    resized_image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode()
    src = f"data:image/jpeg;base64,{encoded_image}"

    return html.Img(src=src, style={'maxWidth': '100%', 'height': 'auto'})

def get_initial_aggregate_data():
    response = aggregate_table.scan()
    aggregate_data = {item['label']: item['count'] for item in response['Items']}
    return aggregate_data

def convert_image_to_pil(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    return image

def upload_to_dynamodb(counter, lon, lat, classes_string):
    try:
        records_table.put_item(
            Item={
                'id': str(counter),
                'datetime': datetime.now().isoformat(),
                'lon': str(lon),
                'lat': str(lat),
                'classes': classes_string,
                'image_url': f'image_{counter}.jpg',
            })
        print("Uploaded data to DynamoDB")
        return True
    except Exception as e:
        print("Couldn't upload data to DynamoDB: ", e)
        return False

upload_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4('Upload Image', className='card-title'),
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                ),
                html.Div(id='output-image-upload')
            ]
        ),
    ], style={'margin': '10px'}
)

bar_chart_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4('Inference Results', className='card-title'),
                dcc.Graph(id='inference-graph', config={'displayModeBar': False}),
            ]
        ),
    ], style={'margin': '10px'}
)

bubble_map_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4('Inference Map', className='card-title'),
                dcc.Graph(id='bubble-map', config={'displayModeBar': False}),
            ]
        ),
    ], style={'margin': '10px'}
)

date_picker = dcc.DatePickerRange(
    id='date-picker-range',
    start_date=date.today() - timedelta(days=7),
    end_date=date.today()
)
download_button = html.Button('Download CSV', id='download-button')
download_link = html.A(
    id='download-link',
    children="Download CSV",  # This text won't be visible but is useful for accessibility
    href='',  # This will be updated dynamically
    download='data.csv',  # Suggested filename for the download
    style={'display': 'none'}  # Hide the link
)

app.layout = html.Div(
    [
        html.Div('Machine Learning for Mitigating Litter',
                 style={'backgroundColor': '#FD7E14', 'color': 'white', 'fontSize': 24, 'padding': 10, 'textAlign': 'center'}),
        dcc.Store(id='aggregate-data', data=dict()),  # Add this line to include data storage in the layout
        dbc.Row([dbc.Col(upload_card, width=12, md=6, style={"padding": "10px"}),
                 dbc.Col(bar_chart_card, width=12, md=6, style={"padding": "10px"})]),
        dbc.Row([dbc.Col(bubble_map_card, width=12, md=12, style={"padding": "10px"}),
                 ]),
        dcc.Interval(
            id='interval-component',
            interval=5 * 1000,  # in milliseconds
            n_intervals=0
        ),

    ], style={'backgroundColor':'#0F0F0F', 'padding':'10px'}
)
def get_file_counter():
    try:
        with open('../counter.txt', 'r') as f:
            count = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        count = 1
    with open('../counter.txt', 'w') as f:
        f.write(str(count + 1))
    return count

def store_aggregate_data_in_dynamodb(aggregate_data):
    with aggregate_table.batch_writer() as batch:
        for label, count in aggregate_data.items():
            batch.put_item(Item={'label': label, 'count': count})

def fetch_coordinates_and_classes():
    response = records_table.scan()
    data = []
    for item in response['Items']:
        if item['lon'] != 'None' and item['lat'] != 'None':
            lon = float(item['lon'])
            lat = float(item['lat'])
            classes = item['classes']
            data.append((lon, lat, classes))
    return data

@app.callback(
    Output('output-image-upload', 'children'),
    Output('inference-graph', 'figure'),
    Output('aggregate-data', 'data'),
    Input('upload-image', 'contents'),
    State('aggregate-data', 'data'),
    prevent_initial_call=True
)



def update_output(contents, aggregate_data):
    if contents is None:
        raise dash.exceptions.PreventUpdate
    aggregate_data = get_initial_aggregate_data()
    # Save to a BytesIO object to use with model.predict
    image = convert_image_to_pil(contents)
    image_byte_arr = io.BytesIO()
    image.save(image_byte_arr, format="JPEG", exif=image.info["exif"])
    counter = get_file_counter()
    # Read EXIF data
    image_exif_io = image_byte_arr.getvalue()
    image_exif_io = io.BytesIO(image_exif_io)
    lat, lon = get_image_coordinates(image_exif_io)

    # Resize image and convert image to numpy array
    image_array = convert_image_to_np_array(contents)

    prediction = model.predict(image_array, confidence=10, overlap=30).json()
    children = parse_contents(contents, prediction)

    if aggregate_data is None:
        aggregate_data = {}


    classes = []
    for obj in prediction['predictions']:
        cls = str(obj['class'])
        classes.append(cls)
        aggregate_data[cls] = aggregate_data.get(cls, 0) + 1     # Update aggregate data for each class

    # Prepare class list string for sheet insertion outside of loop
    classes_string = ', '.join((set(classes)))
    # Insert a new row in the google sheet only once for each image
    records_table.put_item(
        Item={
            'id': str(counter),
            'datetime': datetime.now().isoformat(),
            'lon': str(lon),
            'lat': str(lat),
            'classes': classes_string,
            'image_url': f'image_{counter}.jpg',
        }
    )

    bar_graph = go.Figure(data=[go.Bar(x=list(aggregate_data.keys()), y=list(aggregate_data.values()))])

    bar_graph = go.Figure(data=[
        go.Bar(
            x=list(aggregate_data.keys()),
            y=list(aggregate_data.values()),
            marker_color='rgb(55, 83, 109)'    # change the bar color
        )
    ])
    bar_graph.update_layout(
        title_text='Inference Results',
        title_x=0.5,
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        hovermode='x',
        autosize=True,
        bargap=0.2,
        xaxis=dict(
            title='Labels',
            titlefont=dict(
                size=14,
                color='black'
            ),
            showticklabels=True,
            tickangle=45,
            tickfont=dict(
                size=12,
                color='black'
            ),
            automargin = True,  # Automatically adjust margins

    ),
        yaxis=dict(
            title='Count',
            titlefont=dict(
                size=14,
                color='black',
            ),
            showgrid=False,
            gridcolor='rgb(183,183,183)',
        ),
    )

    image_byte_arr.seek(0)  # Ensure BytesIO object is at the start
    uploaded = upload_to_dynamodb(counter, lon, lat, classes_string)
    if not uploaded:
        print("Failed to upload data to DynamoDB")

    if uploaded:
        print("Image has been uploaded to S3")
    else:
        print("Image upload to S3 failed")

    store_aggregate_data_in_dynamodb(aggregate_data)

    return children, bar_graph, aggregate_data

@app.callback(
    Output('bubble-map', 'figure'),
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=True
)
def update_bubble_map(_):
    coords_and_classes = fetch_coordinates_and_classes()
    lons = [item[0] for item in coords_and_classes]
    lats = [item[1] for item in coords_and_classes]
    classes = [item[2] for item in coords_and_classes]

    if coords_and_classes:
        center_lat = np.mean(lats)  # center to the average latitude
        center_lon = np.mean(lons)  # center to the average longitude
        zoom = 1 / (np.std(lons) + np.std(lats))  # adjust zoom level based on standard deviation

        if np.isnan(zoom) or zoom > 10:
            zoom = 10  # set a maximum limit for zoom level
    else:
        # default values for init or if no coordinates available
        center_lat = 0
        center_lon = 0
        zoom = 1

    bubble_map = go.Figure(
        data=go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode='markers',
            marker=dict(
                size=8,
                opacity=0.8,
                symbol='circle',
                color='blue'
            ),
            text=classes,  # Add the inference classes here
            hoverinfo='text'  # Display the text on hover
        )
    )

    bubble_map.update_layout(
        mapbox=dict(
            accesstoken='pk.eyJ1Ijoid2NuLW1sMiIsImEiOiJjbHJscXEwd3Mwb3R5MnFrNGppamN5end1In0.h-v-dm4qidmGX0G1kWHB-g',  # Replace 'Your_mapbox_token' with your actual Mapbox access token
            center=dict(lat=center_lat, lon=center_lon),  # this line controls the center
            zoom=zoom,  # this line controls zoom
            style='mapbox://styles/wcn-ml2/clrlr2ibo003f01qy7m3gf163'  # Replace with your actual style URL
        ),
        autosize=True,
        hovermode='closest',
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
    )

    return bubble_map


@app.callback(
    Output('download-link', 'href'),
    [Input('download-button', 'n_clicks')],
    [State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date')]
)
def generate_csv(n_clicks, start_date, end_date):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Convert dates to match the format in DynamoDB
    start_date_str = datetime.strptime(start_date, '%Y-%m-%d').isoformat()
    end_date_str = datetime.strptime(end_date, '%Y-%m-%d').isoformat()

    # Query DynamoDB
    response = records_table.scan(
        FilterExpression=Key('datetime').between(start_date_str, end_date_str)
    )
    items = response['Items']

    # Convert to pandas DataFrame
    try:
        df = pd.DataFrame(items)
    except ValueError as e:
        print("Error creating DataFrame:", e)
        # Handle the error appropriately (e.g., return an empty href or a message)
        return ''

    # Convert DataFrame to CSV string
    csv_string_io = io.StringIO()
    df.to_csv(csv_string_io, index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string_io.getvalue())
    print("CSV string created:", csv_string)  # Debug print

    return csv_string


if __name__ == "__main__":
    app.run_server(debug=True)

#working google sheet
#working bucket
#naming convention changed

#update the table working!!
#dynamoDB handling working YESSS!
#bounding box error fixed
#want to update the map
#map now working!