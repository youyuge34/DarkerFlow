import tensorflow as tf
import time
from . import help
from . import flow
from .ops import op_create, identity
from .ops import HEADER, LINE
from .framework import create_framework
from ..dark.darknet import Darknet
import json
import os

class TFNet(object):

	_TRAINER = dict({
		'rmsprop': tf.train.RMSPropOptimizer,
		'adadelta': tf.train.AdadeltaOptimizer,
		'adagrad': tf.train.AdagradOptimizer,
		'adagradDA': tf.train.AdagradDAOptimizer,
		'momentum': tf.train.MomentumOptimizer,
		'adam': tf.train.AdamOptimizer,
		'ftrl': tf.train.FtrlOptimizer,
		'sgd': tf.train.GradientDescentOptimizer
	})

	# imported methods
	_get_fps = help._get_fps
	say = help.say
	train = flow.train
	camera = help.camera
	predict = flow.predict
	return_predict = flow.return_predict
	return_predict_thresh = flow.return_predict_thresh
	to_darknet = help.to_darknet
	build_train_op = help.build_train_op
	load_from_ckpt = help.load_from_ckpt

	def __init__(self, FLAGS, darknet = None):
		self.ntrain = 0

		# it is adapting the way to initial the TFNet from code, instead of from command line.
		if isinstance(FLAGS, dict):
			from ..defaults import argHandler
			newFLAGS = argHandler()
			newFLAGS.setDefaults()
			newFLAGS.update(FLAGS)
			FLAGS = newFLAGS

		self.FLAGS = FLAGS
		if self.FLAGS.pbLoad and self.FLAGS.metaLoad:
			self.say('\nLoading from .pb and .meta')
			self.graph = tf.Graph()
			device_name = FLAGS.gpuName \
				if FLAGS.gpu > 0.0 else None
			with tf.device(device_name):
				with self.graph.as_default() as g:
					self.build_from_pb()
			return

		'''
		此处的darknet仅仅是使用python语言进行解析文件与读入
		转化为darknet's `layer` object
		并未使用tensorflow
		'''
		if darknet is None:
			# 初始化darknet，读入解析cfg文件与weights文件
			darknet = Darknet(FLAGS)
			self.ntrain = len(darknet.layers)

		self.darknet = darknet
		args = [darknet.meta, FLAGS]
		self.num_layer = len(darknet.layers)

		# self.say('meta:',darknet.meta)

		#  the meta after initial the Darknet:
		# {'noobject_scale': 1, 'rescore': 1, 'bias_match': 1, 'random': 1, 'absolute': 1, 'model': 'cfg/yolo.cfg',
		#  'type': '[region]', 'jitter': 0.3, 'thresh': 0.1,
		#  'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
		#  'coords': 4, 'class_scale': 1, 'softmax': 1, 'out_size': [19, 19, 425], 'classes': 80,
		#  'net': {'learning_rate': 0.001, 'width': 608, 'burn_in': 1000, 'channels': 3, 'policy': 'steps',
		#  'exposure': 1.5, 'hue': 0.1, 'decay': 0.0005, 'batch': 1, 'steps': '400000,450000', 'saturation': 1.5,
		#  'subdivisions': 1, 'max_batches': 500200, 'scales': '.1,.1', 'type': '[net]', 'momentum': 0.9, 'angle': 0,
		#  'height': 608}, 'coord_scale': 1, 'inp_size': [608, 608, 3], 'object_scale': 5, 'num': 5}

		# 根据darknet不同的meta构建不同的网络，若meta['type']为'[region]'则构建网络为yolov2
		self.framework = create_framework(*args)
		self.meta = darknet.meta
		# 此时，framework的构造器向meta中添加了labels、label-colors、覆写FLAGS.threshold信息

		# 正式开始构建网络
		self.say('\nBuilding net ...')
		start = time.time()
		self.graph = tf.Graph()
		device_name = FLAGS.gpuName \
			if FLAGS.gpu > 0.0 else None
		with tf.device(device_name):
			with self.graph.as_default() as g:
				# 真正使用tf构建网络
				self.build_forward()
				# 配置GPU,如是train阶段则构建训练op,创建session,初始化全局变量,加载ckpt文件
				self.setup_meta_ops()
		self.say('Finished TFNet building in {}s\n'.format(
			time.time() - start))

	def build_from_pb(self):
		with tf.gfile.FastGFile(self.FLAGS.pbLoad, "rb") as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())

		tf.import_graph_def(
			graph_def,
			name=""
		)
		with open(self.FLAGS.metaLoad, 'r') as fp:
			self.meta = json.load(fp)
		self.framework = create_framework(self.meta, self.FLAGS)

		# Placeholders
		self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
		self.feed = dict() # other placeholders
		self.out = tf.get_default_graph().get_tensor_by_name('output:0')

		self.setup_meta_ops()

	def build_forward(self):
		'''
		真正使用tensorflow向前构造网络
		:return:
		'''
		verbalise = self.FLAGS.verbalise

		# Placeholders, dimension is [None,w,h,c]
		# the parameters w,h,c is in meta['inp_size'] from the '.cfg' file
		inp_size = [None] + self.meta['inp_size']
		self.inp = tf.placeholder(tf.float32, inp_size, 'input')
		self.feed = dict() # other placeholders

		# Build the forward pass
		state = identity(self.inp)
		roof = self.num_layer - self.ntrain
		self.say(HEADER, LINE)
		for i, layer in enumerate(self.darknet.layers):
			scope = '{}-{}'.format(str(i),layer.type)
			args = [layer, state, i, roof, self.feed]
			state = op_create(*args)
			# state.out为这一layer的结果，也为下一layer的input
			mess = state.verbalise()
			self.say(mess)
		self.say(LINE)

		self.top = state
		self.out = tf.identity(state.out, name='output')

	def setup_meta_ops(self):
		'''
		配置GPU
		如是train阶段则构建训练op
		创建session
		初始化全局变量
		加载ckpt文件
		:return:
		'''
		cfg = dict({
			'allow_soft_placement': False,
			'log_device_placement': False
		})

		utility = min(self.FLAGS.gpu, 1.)
		if utility > 0.0:
			self.say('GPU mode with {} usage'.format(utility))
			cfg['gpu_options'] = tf.GPUOptions(
				per_process_gpu_memory_fraction = utility)
			cfg['allow_soft_placement'] = True
		else:
			self.say('Running entirely on CPU')
			cfg['device_count'] = {'GPU': 0}

		if self.FLAGS.train: self.build_train_op()

		if self.FLAGS.summary:
			self.summary_op = tf.summary.merge_all()
			self.writer = tf.summary.FileWriter(self.FLAGS.summary + 'train')

		self.sess = tf.Session(config = tf.ConfigProto(**cfg))
		self.sess.run(tf.global_variables_initializer())

		if not self.ntrain: return
		self.saver = tf.train.Saver(tf.global_variables(),
			max_to_keep = self.FLAGS.keep)

		# 在Darknet的构建中若load是weights/ckpt文件或者没填写，则会将load置0.因此会跳过这一步
		if self.FLAGS.load != 0: self.load_from_ckpt()

		if self.FLAGS.summary:
			self.writer.add_graph(self.sess.graph)

	def savepb(self):
		"""
		Create a standalone const graph def that 
		C++	can load and run.
		"""
		darknet_pb = self.to_darknet()
		flags_pb = self.FLAGS
		flags_pb.verbalise = False

		flags_pb.train = False
		# rebuild another tfnet. all const.
		tfnet_pb = TFNet(flags_pb, darknet_pb)
		tfnet_pb.sess = tf.Session(graph = tfnet_pb.graph)
		# tfnet_pb.predict() # uncomment for unit testing
		name = 'built_graph/{}.pb'.format(self.meta['name'])
		os.makedirs(os.path.dirname(name), exist_ok=True)
		#Save dump of everything in meta
		with open('built_graph/{}.meta'.format(self.meta['name']), 'w') as fp:
			json.dump(self.meta, fp)
		self.say('Saving const graph def to {}'.format(name))
		graph_def = tfnet_pb.sess.graph_def
		tf.train.write_graph(graph_def,'./', name, False)