import numpy as np
import argparse
import time, os
import torchvision
import csv
import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer, AutoModel
from sklearn import metrics
import scipy.io as sio


class ReverseLayerF(Function):

	#@staticmethod
	def forward(ctx, x):
		#self.lambd = args.lambd
		return x.view_as(x)

	#@staticmethod
	def backward(ctx, grad_output):
		return grad_output.neg()
		#return (grad_output * -self.lambd)

def grad_reverse(x):
	return ReverseLayerF().apply(x)



# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
	def __init__(self, args):
		super(CNN_Fusion, self).__init__()

		self.args = args
		self.event_num = args.event_num

		C = args.class_num
		self.hidden_size = args.hidden_dim

		# TEXT 
		self.embed = AutoModel.from_pretrained('bert-base-uncased')
		self.fc1 = nn.Linear(768,self.hidden_size)

		#IMAGE
		resnet50 = torchvision.models.resnet18(pretrained=True)
		for param in resnet50.parameters():
			param.requires_grad = False     
		# visual model
		num_ftrs = resnet50.fc.out_features
		self.resnet = resnet50
		self.image_fc1 = nn.Linear(num_ftrs,  self.hidden_size)


		###social context##################################################
		self.social = nn.Linear(1069, self.hidden_size)
		################################################################################


		## Class  Classifier
		self.class_classifier = nn.Sequential()
		self.class_classifier.add_module('c_fc1', nn.Linear(96, 100))
		self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

		###Event Classifier
		self.domain_classifier = nn.Sequential()
		self.domain_classifier.add_module('d_fc1', nn.Linear(96, 100))
		self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
		self.domain_classifier.add_module('d_fc2', nn.Linear(100, 17))
		self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

	def forward(self, text, image, mask, social):
		### IMAGE #####
		image = image.float() 
		image = self.resnet(image)
		image = F.leaky_relu(self.image_fc1(image))

		_, cls_hs  = self.embed(text, attention_mask=mask, return_dict=False)
		text = F.leaky_relu(self.fc1(cls_hs))

		social = self.social(social)
		social = F.leaky_relu(social)

		text_image_social = torch.cat((text, image, social), 1)

		#print("**************************************************")
		#print(text_image_social.shape)
		#print("**************************************************")
		### Fake or real
		class_output = self.class_classifier(text_image_social)
		#print("**************************************************")

		## Domain (which Event )
		reverse_feature = grad_reverse(text_image_social)
		domain_output = self.domain_classifier(reverse_feature)
		#print("**************************************************")

		return class_output, domain_output

def to_var(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x)


def to_np(x):
	return x.data.cpu().numpy()

def select(train, selec_indices):
	temp = []
	for i in range(len(train)):
		print("length is "+str(len(train[i])))
		print(i)
		#print(train[i])
		ele = list(train[i])
		temp.append([ele[i] for i in selec_indices])
	return temp



def make_weights_for_balanced_classes(event, nclasses = 15):
	count = [0] * nclasses
	for item in event:
		count[item] += 1
	weight_per_class = [0.] * nclasses
	N = float(sum(count))
	for i in range(nclasses):
		weight_per_class[i] = N/float(count[i])
	weight = [0] * len(event)
	for idx, val in enumerate(event):
		weight[idx] = weight_per_class[val]
	return weight



def main(args):
	print('loading data')

	train, validate, test = load_data("./dataset/rawdata_add_event_remove_na.csv")

	train_dataset = Rumor_Data(train)

	validate_dataset = Rumor_Data(validate)

	test_dataset = Rumor_Data(test) 

	train_loader = DataLoader(dataset=train_dataset,
							  batch_size=args.batch_size,
							  shuffle=True)

	validate_loader = DataLoader(dataset = validate_dataset,
								 batch_size=args.batch_size,
								 shuffle=False)

	test_loader = DataLoader(dataset=test_dataset,
							 batch_size=args.batch_size,
							 shuffle=False)

	print('building model')
	model = CNN_Fusion(args)
	
	#print("Size of final output in the forward pass")
	#print(model)
    
	if torch.cuda.is_available():
		print("CUDA")
		model.cuda()

	# Loss and Optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
								 lr= args.learning_rate)
	#optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, list(model.parameters())),
								 #lr=args.learning_rate)
	#scheduler = StepLR(optimizer, step_size= 10, gamma= 1)


	iter_per_epoch = len(train_loader)
	print("loader size " + str(len(train_loader)))
	best_validate_acc = 0.000
	best_test_acc = 0.000
	best_loss = 100
	best_validate_dir = ''
	best_list = [0,0]

	print('training model')
	adversarial = True
	# Train the Model
	for epoch in range(args.num_epochs):

		p = float(epoch) / 100
		#lambd = 2. / (1. + np.exp(-10. * p)) - 1
		lr = 0.001 / (1. + 10 * p) ** 0.75

		optimizer.lr = lr
		#rgs.lambd = lambd
		start_time = time.time()
		cost_vector = []
		class_cost_vector = []
		domain_cost_vector = []
		acc_vector = []
		valid_acc_vector = []
		test_acc_vector = []
		vali_cost_vector = []
		test_cost_vector = []


		for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
			train_text, train_mask,  train_image, train_social, train_labels, event_labels = \
				to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), to_var(train_data[3]), \
				to_var(train_labels), to_var(event_labels)

			# Forward + Backward + Optimize
			optimizer.zero_grad()

			# train_image = train_image.permute(0, 3, 1, 2)
			class_outputs, domain_outputs = model(train_text, train_image.float(), train_mask, train_social.float())

			## Fake or Real loss
			class_loss = criterion(class_outputs, train_labels)
			# Event Loss
			domain_loss = criterion(domain_outputs, event_labels)
			#print("Class Loss")
			#print(class_loss)
			
			#print("Domain Loss")
			#print(domain_loss)
			loss = class_loss + domain_loss
			loss.backward()
			optimizer.step()
			_, argmax = torch.max(class_outputs, 1)

			cross_entropy = True

			if True:
				accuracy = (train_labels == argmax.squeeze()).float().mean()
			else:
				_, labels = torch.max(train_labels, 1)
				accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()
				
			#print(class_loss.data)
			class_cost_vector.append(class_loss.item())
			domain_cost_vector.append(domain_loss.item())
			cost_vector.append(loss.item())
			acc_vector.append(accuracy.item())
			# if i == 0:
			#     train_score = to_np(class_outputs.squeeze())
			#     train_pred = to_np(argmax.squeeze())
			#     train_true = to_np(train_labels.squeeze())
			# else:
			#     class_score = np.concatenate((train_score, to_np(class_outputs.squeeze())), axis=0)
			#     train_pred = np.concatenate((train_pred, to_np(argmax.squeeze())), axis=0)
			#     train_true = np.concatenate((train_true, to_np(train_labels.squeeze())), axis=0)



		model.eval()
		validate_acc_vector_temp = []
		for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):

			validate_text, validate_mask,  validate_image, validate_social, validate_labels, event_labels = \
				to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), to_var(validate_data[3]), \
				to_var(validate_labels), to_var(event_labels)

			validate_outputs, domain_outputs = model(validate_text, validate_image, validate_mask, validate_social.float())
			_, validate_argmax = torch.max(validate_outputs, 1)
			vali_loss = criterion(validate_outputs, validate_labels)
			#domain_loss = criterion(domain_outputs, event_labels)
				#_, labels = torch.max(validate_labels, 1)
			validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
			vali_cost_vector.append( vali_loss.item())
				#validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
			validate_acc_vector_temp.append(validate_accuracy.item())
		validate_acc = np.mean(validate_acc_vector_temp)
		valid_acc_vector.append(validate_acc)
		model.train()
		print ('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
				% (
				epoch + 1, args.num_epochs,  np.mean(cost_vector), np.mean(class_cost_vector),  np.mean(domain_cost_vector),
					np.mean(acc_vector),   validate_acc))

		if validate_acc > best_validate_acc:
			best_validate_acc = validate_acc
			if not os.path.exists(args.output_file):
				os.mkdir(args.output_file)

			best_validate_dir = args.output_file + str(epoch + 1) + '.pkl'
			torch.save(model.state_dict(), best_validate_dir)

		duration = time.time() - start_time
		# print ('Epoch: %d, Mean_Cost: %.4f, Duration: %.4f, Mean_Train_Acc: %.4f, Mean_Test_Acc: %.4f'
		# % (epoch + 1, np.mean(cost_vector), duration, np.mean(acc_vector), np.mean(test_acc_vector)))
		# best_validate_dir = args.output_file + 'weibo_GPU2_out.' + str(52) + '.pkl'
	


	# Test the Model
	print('testing model')
	model = CNN_Fusion(args)
	model.load_state_dict(torch.load(best_validate_dir))
	#    print(torch.cuda.is_available())
	if torch.cuda.is_available():
		model.cuda()
	model.eval()
	test_score = []
	test_pred = []
	test_true = []
	for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
		test_text, test_image, test_mask, test_labels = to_var(
			test_data[0]), to_var(test_data[1]), to_var(test_data[2]), to_var(test_labels)
		test_outputs, domain_outputs= model(test_text, test_image, test_mask)
		_, test_argmax = torch.max(test_outputs, 1)
		if i == 0:
			test_score = to_np(test_outputs.squeeze())
			test_pred = to_np(test_argmax.squeeze())
			test_true = to_np(test_labels.squeeze())
		else:
			test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
			test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
			test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

	test_accuracy = metrics.accuracy_score(test_true, test_pred)
	test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
	test_precision = metrics.precision_score(test_true, test_pred, average='macro')
	test_recall = metrics.recall_score(test_true, test_pred, average='macro')
	test_score_convert = [x[1] for x in test_score]
	test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
	
	test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

	print("Classification Acc: %.4f, AUC-ROC: %.4f"
		  % (test_accuracy, test_aucroc))
	print("Classification report:\n%s\n"
		  % (metrics.classification_report(test_true, test_pred)))
	print("Classification confusion matrix:\n%s\n"
		  % (test_confusion_matrix))


def read_data(csvFile):
	text_context = []
	img_pic =[]
	social_context =[]
	label = []
	event_label = []

	event_list =  ['varoufakis', 'syrianboy', 'garissa', 'nepal', 'samurai', 'columbianChemicals', 'passport',
	'underwater', 'bringback', 'elephant', 'boston', 'sochi', 'sandy', 'malaysia', 'livr', 'pigFish', 'eclipse']

	#label_list = ['pants-fire, false,  barely-true, half-true', 'mostly-true', 'true']


	with open(csvFile, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',')
		for count, row in enumerate(spamreader):

			if count != 0:
				img_list = row[2].split(",")

				for num_img in range(len(img_list)):
					
#                     ************** Text *******************
					text_context.append(row[7])
	
#                     ************** Images *******************                    
					img_name = "".join(['../../dataset/images/',img_list[num_img],'.jpg'])
					if os.path.exists(img_name) == True:
						img_file = cv2.imread(img_name)
						img = cv2.resize(img_file, (224, 224), interpolation=cv2.INTER_CUBIC)/255
					else:
						img = np.zeros((224, 224, 3), dtype = np.float64)
					img_pic.append(img)
					

#                     ************** Social *******************
					num_words_array = np.zeros((30,), dtype=int)
					num_words_array[int(float(row[8]))] = 1
		
					text_length_array = np.zeros((40,), dtype=int)
					q_text_length_array = (int(float(row[9])) - 1)// 5
					text_length_array[q_text_length_array] = 1
					
				
					num_questmark_array = np.zeros((10,), dtype=int)
					num_questmark_array[int(float(row[11]))] = 1
					
					num_exclammark_array = np.zeros((16,), dtype=int)
					num_exclammark_array[int(float(row[13]))] = 1
					
					Social_array = np.concatenate((num_words_array, text_length_array, \
												   num_questmark_array, num_exclammark_array))
					
					if row[14] == False:
						Social_array = np.concatenate((Social_array, [0]))
					else:
						Social_array = np.concatenate((Social_array, [1]))
						
					if row[15] == False:
						Social_array = np.concatenate((Social_array, [0]))
					else:
						Social_array = np.concatenate((Social_array, [1]))
						
					if row[16] == False:
						Social_array = np.concatenate((Social_array, [0]))
					else:
						Social_array = np.concatenate((Social_array, [1]))
					
					if row[17] == False:
						Social_array = np.concatenate((Social_array, [0]))
					else:
						Social_array = np.concatenate((Social_array, [1]))
					
					if row[18] == False:
						Social_array = np.concatenate((Social_array, [0]))
					else:
						Social_array = np.concatenate((Social_array, [1]))
			
					
					num_uppercasechars_array = np.zeros((100,), dtype=int)
					num_uppercasechars_array[int(float(row[19]))] = 1
			
					num_possentiwords_array = np.zeros((10,), dtype=int)
					num_possentiwords_array[int(float(row[20]))] = 1
					
					num_negsentiwords_array = np.zeros((10,), dtype=int)
					num_negsentiwords_array[int(float(row[21]))] = 1
					
					num_hashtags_array = np.zeros((15,), dtype=int)
					num_hashtags_array[int(float(row[22]))] = 1
					
					num_URLs_array = np.zeros((10,), dtype=int)
					num_URLs_array[int(float(row[23]))] = 1
					
					num_friends_array = np.zeros((200,), dtype=int)
					q_num_friends_array = (int(float(row[24])) -1) // 1000
					num_friends_array[q_num_friends_array] = 1

					
					num_followers_array = np.zeros((220,), dtype=int)
					q_num_followers_array = (int(float(row[25])) -1) // 30000
					num_followers_array[q_num_followers_array] = 1
				
					
					folfriend_ratio = float(row[26])
				
					times_listed_array = np.zeros((200,), dtype=int)
					q_times_listed = (int(float(row[27])) -1) // 620
					times_listed_array[q_times_listed] = 1
					
					Social_array = np.concatenate((Social_array, \
												   num_uppercasechars_array, num_possentiwords_array, 
												   num_negsentiwords_array, num_hashtags_array, \
												   num_URLs_array, num_friends_array, \
												   num_followers_array, [folfriend_ratio], \
												   times_listed_array))
					
					if row[28] == False:
						Social_array = np.concatenate((Social_array, [0]))
					else:
						Social_array = np.concatenate((Social_array, [1]))
					
					if row[29] == False:
						Social_array = np.concatenate((Social_array, [0]))
					else:
						Social_array = np.concatenate((Social_array, [1]))
						
					num_posts_array = np.zeros((200,), dtype=int)
					q_num_posts = (int(float(row[30])) -1) // 5650
					num_posts_array[q_num_posts] = 1 
					
					Social_array = np.concatenate((Social_array, num_posts_array))
					
					social_context.append(Social_array)
					
#                     ************** Fake *******************  
					label_array = np.zeros((1,), dtype=int)           
					if row[5] == "fake" or row[5] == 'FALSE' or row[5] == 'false':
						label_ind = 0
					else:
						label_ind = 1
					label.append(int(label_ind))
					
#                     ************** Event *******************   
					#event_array = np.zeros((162,), dtype=int)
					for event in row[6].split(","):
						#event_array[int(event_list.index(event))] = 1
					    event_label.append(int(event_list.index(event)))
					    break

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	tokens_info = tokenizer.batch_encode_plus(
		text_context,
		max_length = 120,
		padding = 'max_length',
		truncation=True,
		return_token_type_ids=False,
		return_attention_mask = True,
		return_tensors = 'pt'
	)

	tokens_post_text = tokens_info['input_ids']
	tokens_mask = tokens_info['attention_mask']
	img_pic = np.asarray(img_pic)
	img_pic = np.transpose(img_pic, (0, 3, 1, 2))
	social_context = np.asarray(social_context)
	label = np.asarray(label)
	event_label = np.asarray(event_label)

	return tokens_post_text, tokens_mask, img_pic, social_context, label, event_label

def load_data(csvFile):

	train = {}
	validate = {}
	test = {}

	tokens_post_text, tokens_mask, img_pic, social_context, label, event_label = read_data(csvFile)
	
	print("Labels")
	print(np.unique(label))
	
	print("Event Labels")
	print(np.unique(event_label))
    
	tokens_post_text_train, tokens_post_text_temp, \
	tokens_mask_train, tokens_mask_temp, \
	img_pic_train, img_pic_temp, \
	social_context_train, social_context_temp, \
	label_train, label_temp, \
	event_label_train, event_label_temp = train_test_split( \
		tokens_post_text, tokens_mask, img_pic, social_context, label, event_label, \
		test_size=0.20, random_state=42)

	tokens_post_text_val, tokens_post_text_test, \
	tokens_mask_val, tokens_mask_test, \
	img_pic_val, img_pic_test, \
	social_context_val, social_context_test, \
	label_val, label_test, \
	event_label_val, event_label_test = train_test_split( \
		tokens_post_text_temp, tokens_mask_temp, img_pic_temp, social_context_temp, \
		label_temp, event_label_temp, \
		test_size=0.20, random_state=42)

	train['post_text'] = tokens_post_text_train
	train['mask'] = tokens_mask_train
	train['img'] = torch.tensor(img_pic_train)
	train['social'] = torch.tensor(social_context_train)
	train['label'] = torch.tensor(label_train)
	train['event'] = torch.tensor(event_label_train)
	
	validate['post_text'] = tokens_post_text_val
	validate['mask'] = tokens_mask_val
	validate['img'] = torch.tensor(img_pic_val)
	validate['social'] = torch.tensor(social_context_val)
	validate['label'] = torch.tensor(label_val)
	validate['event'] = torch.tensor(event_label_val)

	test['post_text'] = tokens_post_text_test
	test['mask'] = tokens_mask_test
	test['img'] = torch.tensor(img_pic_test)
	test['social'] = torch.tensor(social_context_test)
	test['label'] = torch.tensor(label_test)
	test['event'] = torch.tensor(event_label_test)

	return train, validate, test

class Rumor_Data(Dataset):
	def __init__(self, dataset):
		self.text = dataset['post_text']
		self.mask = dataset['mask']
		self.img = dataset['img']
		self.social =dataset['social']
		self.label = dataset['label']
		self.event = dataset['event']

		print('TEXT: %d, Image: %d, labe: %d, Event: %d'
			   % (len(self.text), len(self.img), len(self.label), len(self.event)))

	def __len__(self):
		return len(self.label)

	def __getitem__(self, idx):
		return (self.text[idx], self.mask[idx], self.img[idx], self.social[idx]), \
		self.label[idx], self.event[idx]

def transform(event):
	matrix = np.zeros([len(event), max(event) + 1])
	#print("Translate  shape is " + str(matrix))
	for i, l in enumerate(event):
		matrix[i, l] = 1.00
	return matrix


def parse_arguments(parser):
	parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
	#parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
	parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
	parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

	parse.add_argument('--static', type=bool, default=True, help='')
	parser.add_argument('--sequence_length', type=int, default=28, help='')
	parser.add_argument('--class_num', type=int, default=2, help='')
	parser.add_argument('--hidden_dim', type=int, default = 32, help='')
	parser.add_argument('--embed_dim', type=int, default=32, help='')
	parser.add_argument('--vocab_size', type=int, default=300, help='')
	parser.add_argument('--dropout', type=int, default=0.5, help='')
	parser.add_argument('--filter_num', type=int, default=5, help='')
	parser.add_argument('--lambd', type=int, default= 1, help='')
	parser.add_argument('--text_only', type=bool, default= False, help='')

	#    parser.add_argument('--sequence_length', type = int, default = 28, help = '')
	#    parser.add_argument('--input_size', type = int, default = 28, help = '')
	#    parser.add_argument('--hidden_size', type = int, default = 128, help = '')
	#    parser.add_argument('--num_layers', type = int, default = 2, help = '')
	#    parser.add_argument('--num_classes', type = int, default = 10, help = '')
	parser.add_argument('--d_iter', type=int, default=3, help='')
	parser.add_argument('--batch_size', type=int, default=100, help='')
	parser.add_argument('--num_epochs', type=int, default=100, help='')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='')
	parser.add_argument('--event_num', type=int, default=10, help='')

	#    args = parser.parse_args()
	return parser


if __name__ == '__main__':
	parse = argparse.ArgumentParser()
	parser = parse_arguments(parse)
	train = '' 
	test = ''
	output = './results/twitter/'
	args = parser.parse_args([train, test, output])
	print("here")
	main(args)
   

