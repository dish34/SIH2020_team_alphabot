import nltk
import time
import webbrowser
import random
from nltk.stem.lancaster import LancasterStemmer 
stemmer =LancasterStemmer()

import numpy
import tflearn
import tensorflow 
import random
import json
import pickle

with open("intents.json") as file:
	data = json.load(file)

#try:
	#with open("data.pickle","rb") as f:
	#	words, labels, training, output = pickle.load(f)

#except:
words = []
labels = []
docs_x = []
docs_y = []
for intent in data["intents"]:
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs_x.append(wrds)
		docs_y.append(intent["tag"])
	if intent["tag"] not in labels:
		labels.append(intent["tag"])
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)
training = []
output = []
out_empty = [0 for _ in range(len(labels))]
for x, doc in enumerate(docs_x):
	bag = []

	wrds=[stemmer.stem(w) for w in doc]

	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)

	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1

	training.append(bag)
	output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)
with open("data.pickle","wb") as f:
	pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
#try:
	#model.load("model.tflearn")
#except:
model.fit(training, output , n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

def fun1():
	print("Hi! I am Alphabot. Type to start talking with me(type quit to stop).")
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break
		results = model.predict([bag_of_words(inp, words)])[0]
		results_index = numpy.argmax(results)
		tag = labels[results_index]
		if results[results_index] > 0.7:
			for tg in data["intents"]:
				if tg ['tag'] == tag:
					responses = tg['responses']
			print(random.choice(responses))
		else:
			print("I didn't get you! , try to ask different")

def chat():	 
	PS=["Be AWARE of your surroundings at all times and trust your INSTINCTS.","Stay in well lit, populated pathways. Avoid shortcuts.",
	"Travel in groups. there is always safety in numbers.",
	"Walk with your head upright. Make eye contact. Thieves often target victims who are not paying attention to their surroundings or who are looking down.",
	"Pay attention to your surroundings when using electronics on the streets, subways & buses. do not TUNE yourself out. do not WALK AND TEXT.",
	"Avoid travelling through parks after dark.","Before entering your apartment building, have your keys ready. do not hold doors for anyone whom you do not know.",
	"If you observe anyone acting in a suspicious manner, or if you feel threatened in any way, call the police immediately by dialing 100.",
	"Pay attention to your surroundings, minimize the amount of time you talk on your cell phone in public places.",
	"If you suspect you are being followed, go into an open store and have the clerk call the Police or Public Safety right-away.",
	"Plan your route, know where you are going before leaving and try to walk places with your friends rather than alone.",
	"Stick to well-lighted, well-traveled streets. Avoid shortcuts through wooded areas, parking lots, or alleys.",
	"Take the safest route to and from schools, stores, or your friends houses. Know where to go for help if you need it.",
	"do not display your cash or any other inviting targets like pagers, cell phones, hand-held electronic games, or expensive jewellery and clothing.",
	"Carry your backpack or purse close to your body and keep it closed. Just carrying a wallet? Put it inside your coat or front pants pocket, not in your back pocket or in your backpack.",
	"Have your car or house key in your hand before you reach the door.",
	"If you think someone is following you, change direction or cross the street. If there are still there, move quickly toward an open store or restaurant or a lighted house. do not be afraid to yell for help."]

	DS=["Keep your car in good running condition. Make sure there is enough gas to get where you are going and back.",
	"Turn the ignition off and take your car keys with you, even if you just have to run inside for one minute.",
	"Roll up the windows and lock car doors, even if you are coming right back. Check inside and out before getting in.",
	"Avoid parking in isolated areas. If you are uncomfortable, ask a security guard or store staff to watch you or escort you to your car.",
	"Drive to the nearest gas station, open business, or other well-lighted, crowded area to get help if you think you are being followed. do not head home.",
	"Use your cellular phone, if you have one, to call the police if you are being followed or you have seen an accident. Otherwise, stay off your cellular phone while you are driving.",
	"do not pick up hitchhikers. do not hitchhike.",
	"Keep your car in gear while it is stopped at an intersection.",
	"Keep all doors and windows closed and locked.",
	"Park in well-lighted, designated parking areas.",
	"If you must carry valuables, keep them out of sight in your trunk.",
	"Keep change hidden in your car for emergency telephone calls. Carry a cellular phone for emergencies. Keep an aerosol tire inflator in your car for emergencies.",
	"If your car breaks down, raise the hood, and then stay inside with the doors locked. If someone stops to help, do not open your window or door, or accept a ride. Ask them to call for assistance.",
	"If you see a parked vehicle requiring assistance, do not stop. Go to a telephone and call for aid.",
	"If you are deliberately forced to stop your vehicle, lock the doors, roll up the windows, and sound the horn for help. If you are followed or harassed by someone in another vehicle, drive to a police department, fire station, or open business and seek help. Do not drive into your driveway or park in a deserted area.",
	"If you are followed as you turn into your driveway at night, stay in your car with the doors locked until you identify the occupants of the other car. Sound your horn to get the attention of your neighbors."]

	FS=["Utilize an ATM located inside an open business whenever possible.","Avoid using street ATMs during night time hours.",
	"Always be aware of suspicious persons or vehicles in the area of the ATM. Trust your gut feeling. If things do not feel right, avoid that ATM.",
	"Have ATM card out of your wallet or purse before approaching the ATM.","do not write your ATM personal identification number on your card or keep the number in your wallet.",
	"When entering your personal identification number, try to keep the numbers away from the view of others.","do not withdraw large amounts of cash.",
	"Secure your money at the ATM. do not walk away with money in hand.","Always take your receipt with you.","If a robber demands your money, do not argue or fight with the suspect. Note the robber description and give the robber the money and get away as soon as it is safe to do so. Remember the money is not worth getting hurt over.",
	"do not carry important numbers or passwords with you and do not use your date of birth as your password.","Never leave receipts behind.","Sign your new credit cards immediately.",
	"Report lost or stolen credit cards immediately. Make sure you keep the numbers of the issuers somewhere besides on the back of your card.","Always check your monthly financial statements carefully against your receipts and review your consumer credit report annually.",
	"do not leave mail in your mailbox for more than a day. If you are gone, arrange to have a trusted neighbor or friend pick up your mail.","Shred or tear up all unnecessary documents that have your personal information on them.",
	"Never put in a credit or debit card number through a website unless it offers a secure transaction.A secure transaction will have a padlock icon at the bottom strip of the web page. Also, the URL address will change from http to https on the page where you input personal data.",
	"As soon as you discover your identity is being used, you can begin to fight back to lessen the damage the criminal can do. This is why checking your financial statements frequently and carefully can be your best first step towards discovering an ID theft."]

	HS=["Keep your doors locked and windows covered even in daytime.","Properly check whether the doors and windows of your house are closed and locked when you are going out.","Fix grills on windows and glass panelled doors.",
	"Keep side doors pad locked and main door bolted.","Inform your trusted neighbours only about your absence from your home. Provide lights on the exit points of your house.",
	"Do not keep a large amount of cash and ornaments in the house.","Lock doors and windows. Do so not only while everyones at school or work, but while you are at home. Use good deadbolt locks on all doors, and consider buying security doors to add a layer of protection (and potentially a bug screen!).",
	"Invest in a home security system. Options include one monitored by a central service, which will contact you and refer the call to the police, or simply one that sounds a siren when triggered. (The usual triggers are that a door is opened, a window broken, or a motion sensor set off.) The latter type of system is obviously more DIY, but the truth is that most professional burglars will be in and out of your house in five minutes anyway, so as to never meet the police",
	"Turn on outside lights from dusk until dawn. At least one light in front or your house and one in back is ideal, plus lights to cover any dark areas. Criminals like the cover of darkness. You do not necessarily have to hit the switch yourself.",
	"Trim shrubs, bushes, and trees in front of or near windows. That way, intruders cannot use the foliage to hide behind before entering your home.",
	"Turn on inside lights when out for the evening. Timers for lamps are easy to use, and inexpensive. A bathroom light is good to leave on, because it is a room that might plausibly be in use at any hour of the day or night. Also, leave a radio or television turned on. A thief who thinks someone is at home is far less likely to break in.",
	"do not leave your garage door opener or car keys in an obvious place within the house. A thief could easily steal these for later use. Also, do not leave anything visible on your car seats if you park outdoors, or any valuables in your car at all, such as laptops, cell phones, or sports equipment.",
	"do not keep jewellery in an obvious jewellery box. it is one of the first things a burglar will grab. Better to buy a small, fireproof safe, or hide things in disguised boxes. But do not put valuables between your mattresses or in your clothing drawers. These are other places burglars head straight for; along with medicine cabinets, in search of prescription drugs that have resale value.",
	"Think twice about using tools and websites or posting messages and photos that inform people of where you are. You may be telling a thief you are not at home, and possibly would not be for weeks.","15.	Stop newspaper and mail deliveries when going out of town. Ask a trusted neighbor or friend to keep an eye on things and to remove any random flyers on and around your front door. Also, contact your local police department and ask if they have a program where officers drive by your home to make sure it is secure.",
	"When placing your name on mailboxes or on your bell, use only the last name, e.g., The Sharma"]

	print("  Welcome to Alphabot  ")
	while(True):
		print("\n1) Crime Awareness\n2) Crime Registration\n3) Do you want to talk with Alphabot?\n4) Exit")
		n=int(input("\nPress the appropriate number for the given details: "))
		if n==1:
			while(True):
				print('''\n1) Personal safety\n2) Driving Safety\n3) Financial Safety\n4) Home Safety\n5) Back to Home Page''')
				r=int(input("\nPress the appropriate number for the given details: "))
				if(r==1):
					random_num=random.choice(PS)
					print(random_num)
				elif(r==2):
					random_num=random.choice(DS)
					print(random_num)
				elif(r==3):
					random_num=random.choice(FS)
					print(random_num)
				elif(r==4):
					random_num=random.choice(HS)
					print(random_num)
				elif(r==5):
					break
		elif n==2:
			while(True):
				print('''\n1) Vehicle Reports\n2) Missing Reports\n3) Complaint\n4) Right To Information\n5) Back to Home Page''')
				r=int(input("\nPress the appropriate number for the given details: "))
				if(r==1):
					while(True):
						print('''\nPress the appropriate number for the given details:\n1) MV Theft eFIR\n2) Stolen Vehicle Search\n3) Unclaimed/Seizd Vehicle Search\n4) Back to Previous Menu''')
						t=int(input("\nPress the appropriate number for the given details: "))
						if(t==1):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://mvt.delhipolice.gov.in/')
						elif(t==2):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://zipnet.in/index.php?page=stolen_vehicles_search&criteria=search')
						elif(t==3):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://zipnet.in/index.php?page=un_identified_vehicles_search&criteria=search')
						elif(t==4):
							break
				elif(r==2):
					while(True):
						print('''\n1) Missing Person Report\n2) Lost & Found\n3) View Complaint/Missing Report Status\n4) Missing/Stolen Mobile Phones\n5)Missing Person Search\n6) Missing Children\n7) Un-Identified Persons Found\n8) Un-Identified Child Found\n9) Un-Identified Dead-bodies\n10) Proclaimed Offender\n11) Arrested Persons\n12) Back to Previous Menu''')
						t=int(input("\nPress the appropriate number for the given details: "))
						if(t==1):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://59.180.234.21:8080/citizenservices/missingpersonregistration.htm')
						elif(t==2):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('https://lostfound.delhipolice.gov.in/')
						elif(t==3):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://59.180.234.21:8080/citizen/complaintSearchOptions.htm')
						elif(t==4):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://www.zipnet.in/index.php?page=missing_mobile_phones')
						elif(t==5):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://www.zipnet.in/index.php?page=mps_report_missing_2&criteria=search')
						elif(t==6):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('https://trackthemissingchild.gov.in/trackchild/index.php')
						elif(t==7):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://www.zipnet.in/index.php?page=mps_report_4&criteria=search')
						elif(t==8):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://www.zipnet.in/index.php?page=mps_report_8&criteria=search')
						elif(t==9):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://www.zipnet.in/index.php?page=mps_report_2')
						elif(t==10):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://www.delhipolice.nic.in/proclaimed-eco-off.html')
						elif(t==11):
							print("Find the Arrested Person Details in the attached PDF document that is going to be downloaded")
							time.sleep(5)
							webbrowser.open('http://59.180.234.21:8080/citizenservices/arrestPersonDetailReport.htm')		
						elif(t==12):
							break
				if(r==3):
					while(True):
						print('''\n1) Complaint Lodging\n2) Theft eFIR\n3) View FIR\n4) Back to Previous Menu''')
						t=int(input("\nPress the appropriate number for the given details: "))
						if(t==1):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://59.180.234.21:8080/citizenservices/login.htm')
						elif(t==2):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://propertytheft.delhipolice.gov.in/')
						elif(t==3):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://www.delhipolice.nic.in/view-fir.html')
						elif(t==4):
							break
				elif(r==4):
					while(True):
						print('''\n1) Police Clearance Certificate (PCC)\n2) Character Verification Report (CVR)\n3) Domestic Help Registration (DHR)\n4) Tenant Registration (TR)\n5) Right To Information (RTI)\n6) NOC/License\n7) Wanted Criminals\n8) Back to Previous Menu''')
						t=int(input("\nPress the appropriate number for the given details: "))
						if(t==1):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://pcc.delhipolice.gov.in/')
						elif(t==2):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://cvr.delhipolice.gov.in/')
						elif(t==3):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://59.180.234.21:8080/citizenservices/login.htm;jsessionid=6064c2e7789e302ae86f50e4e34a')
						elif(t==4):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://59.180.234.21:8080/citizenservices/login.htm')
						elif(t==5):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://www.delhipolice.nic.in/rti-main.html')
						elif(t==6):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('https://delhipolicelicensing.gov.in/')
						elif(t==7):
							print("Please fill in the form that you are been redirected to!")
							time.sleep(5)
							webbrowser.open('http://www.delhipolice.nic.in/wanted.html')
						elif(t==8):
							break
				elif(r==5):
					break
		elif n==3:
			fun1()	
		elif n==4:
			print("Thanks for visiting. Have a good Day!!")
			break
chat()
