# Digital-Twin
AI-Digital Twin /ML/DeepLearning

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# IoT data

time = np.linspace(0, 10, 100)
sensor_temp = 25 + 3 * np.sin(time)
flow_rate = 10 + 2 * np.sin(time)

# Physics based

def Physics_model(sensor_temp, flow_rate):
  return flow_rate * (sensor_temp - 25)

  # ML prediction

X = np.array(sensor_temp).reshape(-1, 1)
y = np.array(flow_rate)
model = LinearRegression().fit(X, y)

def ml_prediction(sensor_temp):
  return model.predict(np.array(sensor_temp).reshape(-1, 1))

# dashboard

app = dash.Dash(__name__)

app.layout = html.Div([
     html.H1("Digital Twin Dashboard", style={'color': 'red'}),
    html.P("This is a dashboard that shows the current state of                 the digital twin.", style={'color': 'red'}),
    dcc.Graph(id='temperature-plot'),
    dcc.Graph(id='flow-rate-plot'),
    dcc.Graph(id='physics-model-plot'),
    dcc.Graph(id='ml-prediction-plot'),
    dcc.Interval(
          id='interval-component',
          interval=1*1000, # in milliseconds
          n_intervals=0
        )
    )

@app.callback(
    Output('temperature-plot', 'figure'),
    Output('flow-rate-plot', 'figure'),
    Output('physics-model-plot', 'figure'),
    Output('ml-prediction-plot', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graphs(n):
  sim_temp = Physics_model(sensor_temp, flow_rate) # Corrected typo
  ml_temp = ml_prediction(sensor_temp)

  mae_sim = np.mean(np.abs(sim_temp - flow_rate))
  mae_ml = np.mean(np.abs(ml_temp - flow_rate))

  # Create individual figures for each output
  fig_temp = go.Figure(data=[go.Scatter(x=time, y=sensor_temp, mode='lines', name='Sensor Temperature')])
  fig_temp.update_layout(title='Sensor Temperature', xaxis_title='Time', yaxis_title='Temperature')

  fig_flow = go.Figure(data=[go.Scatter(x=time, y=flow_rate, mode='lines', name='Flow Rate')])
  fig_flow.update_layout(title='Flow Rate', xaxis_title='Time', yaxis_title='Flow Rate')

  fig_physics = go.Figure(data=[go.Scatter(x=time, y=sim_temp, mode='lines', name=f'Physics Model (MAE: {mae_sim:.2f})')])
  fig_physics.update_layout(title='Physics Model Prediction', xaxis_title='Time', yaxis_title='Value')

  fig_ml = go.Figure(data=[go.Scatter(x=time, y=ml_temp, mode='lines', name=f'ML Prediction (MAE: {mae_ml:.2f})')])
  fig_ml.update_layout(title='ML Model Prediction', xaxis_title='Time', yaxis_title='Value')

  return fig_temp, fig_flow, fig_physics, fig_ml

  # run

if __name__ == '__main__':
    app.run(debug=True)
