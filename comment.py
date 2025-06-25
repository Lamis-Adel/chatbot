import nltk
nltk.download('punkt')  # تحميل مجموعة البيانات اللازمة لتقسيم النصوص
from nltk.stem.lancaster import LancasterStemmer  # استيراد أداة تقطيع الكلمات
import numpy as np  # استيراد مكتبة NumPy للعمل مع المصفوفات
import torch  # استيراد PyTorch
import torch.nn as nn  # استيراد مكتبة الشبكات العصبية
import torch.optim as optim  # استيراد أدوات تحسين الشبكات
import random  # استيراد مكتبة العشوائية
import json  # استيراد مكتبة JSON لتحميل البيانات
import pickle  # استيراد مكتبة pickle لتخزين البيانات
from time import sleep  # استيراد دالة sleep للتأخير بين الردود

stemmer = LancasterStemmer()  # إنشاء كائن من نوع LancasterStemmer لتقطيع الكلمات

# تحميل البيانات من ملف intents.json
with open("intents.json") as file:
    data = json.load(file)  # تحميل البيانات JSON إلى متغير data

# معالجة البيانات
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)  # محاولة تحميل البيانات المخزنة مسبقاً
except:
    words = []  # قائمة لتخزين جميع الكلمات
    labels = []  # قائمة لتخزين جميع العلامات الفريدة
    docs_x = []  # قائمة لتخزين الأنماط المفككة
    docs_y = []  # قائمة لتخزين العلامات المرتبطة بالأنماط
    
    # معالجة كل نية في البيانات
    for intent in data["intents"]:
        # معالجة كل نمط متعلق بالنية
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)  # تقسيم النمط إلى كلمات
            words.extend(wrds)  # إضافة الكلمات إلى قائمة words
            docs_x.append(wrds)  # إضافة قائمة الكلمات إلى docs_x
            docs_y.append(intent["tag"])  # إضافة العلامة إلى docs_y

        # إضافة العلامة إلى labels إذا لم تكن موجودة مسبقاً
        if intent["tag"] not in labels:
            labels.append(intent["tag"])  # إضافة العلامة إلى قائمة labels

    # معالجة الكلمات: تقطيعها وتحويلها إلى شكل موحد
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]  # تقطيع الكلمات وتحويلها إلى أحرف صغيرة
    words = sorted(list(set(words)))  # إزالة التكرارات وترتيب الكلمات

    labels = sorted(labels)  # ترتيب العلامات

    training = []  # قائمة لتخزين بيانات التدريب
    output = []  # قائمة لتخزين المخرجات

    out_empty = [0 for _ in range(len(labels))]  # قائمة فارغة بنفس حجم labels

    # إنشاء بيانات التدريب
    for x, doc in enumerate(docs_x):
        bag = []  # قائمة لتمثيل الكيس من الكلمات

        wrds = [stemmer.stem(w) for w in doc]  # تقطيع الكلمات في الوثيقة

        # إنشاء تمثيل الكيس من الكلمات
        for w in words:
            bag.append(1 if w in wrds else 0)  # إضافة 1 إذا كانت الكلمة موجودة، وإلا 0

        output_row = out_empty[:]  # نسخة من القائمة الفارغة
        output_row[labels.index(docs_y[x])] = 1  # تعيين القيمة 1 لمكان العلامة الصحيحة

        training.append(bag)  # إضافة تمثيل الكيس إلى بيانات التدريب
        output.append(output_row)  # إضافة المخرجات إلى قائمة المخرجات

    training = np.array(training)  # تحويل بيانات التدريب إلى مصفوفة NumPy
    output = np.array(output)  # تحويل المخرجات إلى مصفوفة NumPy

    # تخزين البيانات المعالجة للاستخدام المستقبلي
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)  # تخزين الكلمات، العلامات، بيانات التدريب والمخرجات

# نموذج PyTorch (شبكة عصبية)
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()  # استدعاء الباني من الفئة الأم
        self.fc1 = nn.Linear(input_size, hidden_size)  # الطبقة الأولى
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # الطبقة الثانية
        self.fc3 = nn.Linear(hidden_size, output_size)  # الطبقة الثالثة
        self.softmax = nn.Softmax(dim=1)  # دالة softmax لتحويل المخرجات إلى احتمالات

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # تطبيق دالة تفعيل ReLU
        x = torch.relu(self.fc2(x))  # تطبيق دالة تفعيل ReLU
        x = self.fc3(x)  # الطبقة الأخيرة
        return self.softmax(x)  # إرجاع النتائج بعد تطبيق softmax

# إعداد الأبعاد للنموذج
input_size = len(training[0])  # عدد ميزات المدخلات
hidden_size = 8  # عدد وحدات الطبقة المخفية
output_size = len(output[0])  # عدد مخرجات النموذج

# إنشاء نموذج الدردشة
model = ChatbotModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # دالة خسارة الانتروبيا المتقاطعة
optimizer = optim.Adam(model.parameters(), lr=0.001)  # محسن Adam

# التدريب
try:
    model.load_state_dict(torch.load("model.pth"))  # محاولة تحميل النموذج المخزن
except:
    # التدريب على النموذج
    for epoch in range(1000):
        inputs = torch.tensor(training, dtype=torch.float32)  # تحويل بيانات التدريب إلى Tensor
        labels_tensor = torch.tensor(np.argmax(output, axis=1), dtype=torch.long)  # تحويل المخرجات إلى Tensor

        optimizer.zero_grad()  # إعادة تعيين التدرجات
        outputs = model(inputs)  # تمرير البيانات عبر النموذج
        loss = criterion(outputs, labels_tensor)  # حساب الخسارة
        loss.backward()  # حساب التدرجات
        optimizer.step()  # تحديث الأوزان
        
        # طباعة الخسارة كل 100 تكرار
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), "model.pth")  # تخزين النموذج المدرب

# دالة لتحويل الجملة إلى تمثيل كيس الكلمات
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]  # قائمة فارغة بنفس حجم الكلمات
    s_words = nltk.word_tokenize(s)  # تقسيم الجملة إلى كلمات
    s_words = [stemmer.stem(word.lower()) for word in s_words]  # تقطيع الكلمات وتحويلها إلى أحرف صغيرة

    # إنشاء تمثيل الكيس
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1  # تعيين 1 إذا كانت الكلمة موجودة
            
    return np.array(bag, dtype=np.float32)  # إرجاع المصفوفة كنوع float32

# دالة الدردشة
def chat():
    print("Hi, how can I help you?")  # رسالة ترحيبية
    while True:
        inp = input("You: ")  # قراءة إدخال المستخدم
        if inp.lower() == "quit":  # الخروج إذا كتب المستخدم "quit"
            break

        with torch.no_grad():  # عدم حساب التدرجات
            input_bag = torch.tensor(bag_of_words(inp, words), dtype=torch.float32).unsqueeze(0)  # تحويل الإدخال إلى Tensor
            results = model(input_bag).numpy()  # الحصول على المخرجات من النموذج
        
        print("Results:", results)  # طباعة النتائج للتحقق
        results_index = np.argmax(results, axis=1)  # الحصول على الفهرس الأكثر احتمالية
        tag = labels[results_index[0]]  # الحصول على العلامة المرتبطة

        if results[0][results_index[0]] > 0.8:  # إذا كانت النتيجة أكبر من 0.8
            for tg in data["intents"]:
                if tg['tag'] == tag:  # البحث عن الرد المناسب
                    responses = tg['responses']  # الحصول على الردود
            sleep(3)  # تأخير قبل الرد
            Bot = random.choice(responses)  # اختيار رد عشوائي
            print(Bot)  # طباعة الرد
        else:
            print("I don't understand!")  # إذا لم يفهم المدخل

chat()  # بدء الدردشة