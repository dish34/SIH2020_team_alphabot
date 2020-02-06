from flask import Flask, render_template, request, redirect, url_for
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer

app = Flask(__name__)
bot = ChatBot(
    "Candice",
    read_only = True,
    logic_adapters=[
        "chatterbot.logic.BestMatch"
    ]
    )
# chatbot = ChatBot('Training Example')
trainer = ListTrainer(bot)

# trainer.train([
#     "Hello",
#     "Hi there!",
#     "How are you doing?",
#     "I'm doing great.",
#     "That is god to hear",
#     "Thank you.",
#     "You're welcome."
# ])
# trainer.train([
#     "cya", "See you later", "Goodbye", "I am Leaving","bye","Have a Good day",
#     "Sad to see you go :(", "Talk to you later", "Goodbye!"
# ])
# trainer.train([
#     "tell me about yourself","I am an AI bot created by the Alphabot team","who created you?","Alphabot created me for registration of crime online","team that builds you"
# ])
# trainer.train([
#     "i want to register a complaint","Yah sure?","register a complaint","Response to the following questions to file a complaint","register fir","Sure, follow along with me"
# ])
# trainer.train([
#     "Hello",
#     "Hi there!",
#     "How are you doing?",
#     "I'm doing great.",
#     "That is god to hear",
#     "Thank you.",
#     "You're welcome."
# ])
# trainer.train([
#     "Hello",
#     "Hi there!",
#     "How are you doing?",
#     "I'm doing great.",
#     "That is god to hear",
#     "Thank you.",
#     "You're welcome."
# ])
# trainer.train([
#     "Hello",
#     "Hi there!",
#     "How are you doing?",
#     "I'm doing great.",
#     "That is god to hear",
#     "Thank you.",
#     "You're welcome."
# ])

# trainer.train(conversation)

# trainer.train([
#     "Hi there!",
#     "Hello",
# ])

# trainer.train([
#     "Greetings!",
#     "Hello",
# ])

# trainer = ChatterBotCorpusTrainer(bot)

# trainer.train(
#      "chatterbot.corpus.english",
# )
# trainer.train(
#     "./data/greetings_corpus/custom.corpus.json",
#     "./data/my_corpus/"
# )


trainer.train([
   "Hi", "Hi there, how can I help?", "hi", "Hi there, how can I help?","How are you", "Hi there, how can I help?", "Is anyone there?", "Hi there, how can I help?", "Hello", "Hi there, how can I help?", "Good day", "Hi there, how can I help?", "Whats up", "Hi there, how can I help?","good morning", "Hi there, how can I help?","good afternoon", "Hi there, how can I help?","hey", "Hi there, how can I help?"
])
trainer.train([
   "cya","Goodbye!", "See you later","Goodbye!", "Goodbye","Goodbye!", "I am Leaving","Goodbye!","bye","Goodbye!","Have a Good day","Goodbye!","thanks Bye","Goodbye!","see ya","Goodbye!","stop","Goodbye then."
])
trainer.train([
    "tell me about yourself","I am an AI bot created by the Alphabot team","who created you?","Alphabot created me for registration of crime online","team that builds you","Alphabot built me.","owner","Alphabot built me.","who is your owner","Alphabot built me."
])
trainer.train([
    "how old", "I am still Developing", "how old is Alphabot", "I am still Developing", "what is your age", "I am still Developing", "how old are you", "I am still Developing", "age?", "I am still Developing"
]) 
trainer.train([
     "what is your name","I'm Alphabot!", "what should I call you","I'm Alphabot!", "whats your name?","I'm Alphabot!","name","I'm Alphabot!"
])
trainer.train([
   "help me with","Sure, come along","help me","Sure, come along","how can i fill a complaint?","Sure, come along","what can you do","Sure, come along","help me filling the fir","Sure, come along","fir","Sure, come along","terrorism","assault","poaching","tell me about rape cases","trafficking","tell me about rape","i want to know about crime","tell me about sexual assault","child pornography","child sexually abuse","bullying","stalking","cyber grooming","fraud jobs","online sextortion","sexting","smshing","sim swap scam","debit or credit card fraud","impersonation and identity theft","phishing","spamming","virus worms trojans","data breach","pharming","cryptojacking","online drug trafficking","i want to register a complaint","register a complaint","register fir","file my complaint","i want to file a fir"
])
trainer.train([
"when are you guys open","I am available 24*7 and always happy to help you!","what are your hours","I am available 24*7 and always happy to help you!","hours of operation","I am available 24*7 and always happy to help you!","time of work","I am available 24*7 and always happy to help you!","when are you available","I am available 24*7 and always happy to help you!","available hours","I am available 24*7 and always happy to help you!","working hours","I am available 24*7 and always happy to help you!"
     
])
trainer.train([
    "arse","Stop there, this is an Oficial Bot you can't say such thing.","ass","Stop there, this is an Oficial Bot you can't say such thing.","asshole","Stop there, this is an Oficial Bot you can't say such thing.","bastard","Stop there, this is an Oficial Bot you can't say such thing.","bitch","Stop there, this is an Oficial Bot you can't say such thing.","bollocks","Stop there, this is an Oficial Bot you can't say such thing.","bugger","Stop there, this is an Oficial Bot you can't say such thing.","child fucker","Stop there, this is an Oficial Bot you can't say such thing.","christ on a bike","Stop there, this is an Oficial Bot you can't say such thing.","crap","Stop there, this is an Oficial Bot you can't say such thing.","cunt","Stop there, this is an Oficial Bot you can't say such thing.","damn","Stop there, this is an Oficial Bot you can't say such thing.","effing","Stop there, this is an Oficial Bot you can't say such thing.","fuck","Stop there, this is an Oficial Bot you can't say such thing.","frigger","Stop there, this is an Oficial Bot you can't say such thing.","goddamn","Stop there, this is an Oficial Bot you can't say such thing.","godsdamn","Stop there, this is an Oficial Bot you can't say such thing.","hell","Stop there, this is an Oficial Bot you can't say such thing.","holy shit","Stop there, this is an Oficial Bot you can't say such thing.","horseshit","Stop there, this is an Oficial Bot you can't say such thing.","jesus fuck","Stop there, this is an Oficial Bot you can't say such thing.","jesus H. Christ","Stop there, this is an Oficial Bot you can't say such thing.","judas priest","Stop there, this is an Oficial Bot you can't say such thing.","motherfucker","Stop there, this is an Oficial Bot you can't say such thing.","mother fucker","Stop there, this is an Oficial Bot you can't say such thing.","nigga","Stop there, this is an Oficial Bot you can't say such thing.","nigger","Stop there, this is an Oficial Bot you can't say such thing.","prick","Stop there, this is an Oficial Bot you can't say such thing.","shit","Stop there, this is an Oficial Bot you can't say such thing.","slut","Stop there, this is an Oficial Bot you can't say such thing.","son of a bitch","Stop there, this is an Oficial Bot you can't say such thing."
     
])
trainer.train([
    "aha","Please say something relevant.","ahem","Please say something relevant.","bam","Please say something relevant.","boo","Please say something relevant.","blah","Please say something relevant.","bravo","Please say something relevant.","ha ha","Please say something relevant.","hmm","Please say something relevant.","phew","Please say something relevant.","ugh","Please say something relevant.","lol","Please say something relevant."
     
])
trainer.train([
   "who is prime minister","I am not made for this kind of stuff! You should Google this instead.","tell me a joke","I am not made for this kind of stuff! You should Google this instead.","who will win the final","I am not made for this kind of stuff! You should Google this instead.","salman khan","I am not made for this kind of stuff! You should Google this instead.","modi ji","I am not made for this kind of stuff! You should Google this instead.","donald trump","I am not made for this kind of stuff! You should Google this instead.","engineering","I am not made for this kind of stuff! You should Google this instead.","today's weather","I am not made for this kind of stuff! You should Google this instead.","tell me something","I am not made for this kind of stuff! You should Google this instead.","who is alan turing","I am not made for this kind of stuff! You should Google this instead."  
])
trainer.train([
   "you did great","Thank you","nice work","Thank you","that's nice","Thank you","thanks for the help","Thank you","thanks","Thank you","thanx","Thank you"  
])
trainer.train([
   "fire brigade number","Please call 101","fire number","Please call 101","fire department number","Please call 101","call fire department","Please call 101","what is the fire departmentt number","Please call 101"  
])
trainer.train([
    "what is the helpline number","Please call 100","helpline number","Please call 100","helpline no.","Please call 100","police number","Please call 100", "police no.","Please call 100","call police","Please call 100"
])
trainer.train([
    "ambulance number","Please call 108","ambulance call","Please call 108","call ambulance","Please call 108","call hopital","Please call 108","what is the ambulance number","Please call 108"
])
trainer.train([
   "what is the women helpline number","Please call 1090","what is the women's helpline number","Please call 1090","helpline no. for women","Please call 1090","call women's helpline","Please call 1090","women's helpline contact number","Please call 1090"  
])
trainer.train([
  "what is the control room number","Please call 1090","control room number","Please call 1090","control room no.","Please call 1090","call control room","Please call 1090"  
])

@app.route("/")
def home():    
    return render_template("home.html") 
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg') 
    crimereg = str(bot.get_response(userText))
    if crimereg == 'Sure, follow along with me':
        return redirect(url_for('register'))
    return str(bot.get_response(userText)) 
@app.route("/register")
def register():
    return render_template("register.html")
if __name__ == "__main__":    
    app.run()
