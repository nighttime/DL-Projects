class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def viz_model(model, name='viz.png'):
	from keras.utils import plot_model
	plot_model(model, show_shapes=True, to_file=name)
	print('generated chart of the model')