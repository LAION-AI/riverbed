# note: this list is used in conjunction with the flagged words. So words of a sexual nature + words below MIGHT indicate child sexual abuse materials (CSAM).
# see https://www.researchgate.net/publication/259179064_Comparing_Methods_for_Detecting_Child_Exploitation_Content_Online for common CSAM words
# http://antipaedo.lip6.fr/T12/keywords_cnrs.pdf - for top 20 from table 7 and 8 of the paper, plus other similar words , ignoring stopwords like "tuesday"
# WARNING: Translations are probably not accurate and the words, and the original words from the above papers may be out of date. And the list below is incomplete.
banned_words = {
      'ar': {"فتاة صغيرة" , "فتاة صغيرة" , "فتى صغير", "مراهق" , "لولي" , "لولا" , "لوليتا" , "لولي" , "لولي" , "لوليتاغوي" , "فتاة" , "صبي" ,"طفل" , "حمام" },
      'ur': {'نوجوان لڑکی', 'چھوٹی لڑکی', 'چھوٹا لڑکا', 'نوجوان لڑکا','نوعمر','لولی','لولا','لولیتا','لولی','لولی','لولیتاگو','لڑکی','لڑکا','بچہ','بچہ','غسل', 'غسل', 'غسل'},
      'bn': {'ছোট মেয়ে', 'ছোট মেয়ে', 'ছোট ছেলে', 'কিশোর', 'ললি', 'লোলা', 'ললিতা', 'ললি', 'লোলি', 'ললিটাগুই', 'মেয়ে', 'ছেলে' , 'শিশু', 'স্নান',},
      'hi': {'युवा लड़की', 'छोटी लड़की', 'छोटा लड़का', 'किशोर', 'लॉली', 'लोला', 'लोलिता', 'लल्ली', 'लोली', 'लोलितागुय', 'लड़की', 'लड़का' , 'बच्चा', 'स्नान',},
      'eu': {'neska gaztea', 'neska txikia', 'mutil txikia', 'nerabea', 'neska', 'mutil' , 'haurra', 'bainua',},
      'ca': {'noia', 'nena', 'nen petit', 'nen' , 'nen', 'banyer',},
      'vi': {'thiếu niên', 'cậu nhỏ', 'cậu bé', 'cô gái trẻ', 'cô bé', 'cậu bé', 'trẻ em',},
      'zh': {'儿童','女孩','孩子', '小女孩', '小姑娘','小男孩', '年輕女孩','年轻女孩','年轻的女孩','洗澡','洛丽塔','洛麗塔','浴','男孩','萝拉','萝莉','萝莉塔吉','蘿拉','蘿莉','蘿莉塔','青少年'},
      'fr': {'jeune fille','petite fille','petit garçon','ado',  'fille', 'garçon' , 'enfant', 'bain',},
      'id': {'gadis muda','gadis kecil','anak laki-laki kecil','remaja',  'perempuan', 'laki-laki' , 'anak', 'mandi',},
      'fa': {'دختر جوان',  'دختر کوچولو',  'پسر کوچولو',  'نوجوان',  'لولی',  'لولا',  'لولیتا',  'لولی',  'لولی',  'لولیتاگو',  'دختر',  'پسر' ,'کودک',  'حمام', },
      'es': {'niña',  'niño', 'adolescente',},
      'fi': {"tyttö", "poika", "teini", "kylpyhuone",},
      'ja': {'「女の子」','「男の子」','「ティーンエイジャー」', '「バスルーム」'},
      'pt': {'menina', 'menino', 'adolescente', 'pirulito',  'criança', 'banho',},
      'ig': {'nwa agbọghọ', 'nwa agbọghọ', 'nwa agbọghọ',' iri na ụma', 'nwa agbọghọ', 'nwoke' , 'nwa', },
      'sw': {'msichana mdogo','msichana mdogo','kijana mdogo', 'mtoto', 'kuoga',},
      'yo': {'kekere', 'omobinrin', 'omokunrin', 'ọmọ', 'wẹwẹ',},
      'xh': {'intombazana encinci', 'intsha', 'umntwana', 'hlamba', 'inkwenkwe', },
      'zu': {'intombazane', 'intsha', 'intombazane encane',  'umfana omncane','geza', 'ingane', 'yomfana'},
      'default': {'young girl', 'little girl','little boy', 'young boy', 'teen', 'lolli', 'lola', 'lolita', 'lolly', 'loli', 'lolitaguy', 'child', 'kid',  \
                  'incest', "pedo", 'nymphet', 'nimphet', 'babyj', 'voglia', 'eurololita', 'hussyfan', 'kidzilla', 'raygold', 'ygold', 'qwerty', 'qqaazz', 'ptsc', \
                  'pthc', 'tanta', 'mylola', 'arina', 'newstar', 'playtoy', 'imouto', 'lourinha', 'amateurz', \
                  'babyshivid', 'shiori', 'tvg', 'chiharu','kidzilla', 'izzy', 'rika', 'kdquality', 'cbaby', 'nablot', 'lso',  'kinderficker', \
                  '1 yo',  '1 yr',   '2 yo',  '2 yr',   '3 yo',  '3 yr',   '4 yo',  '4 yr',   '5 yo',  '5 yr',   
                  '6 yo',  '6 yr',   '7 yo',  '7 yr',   '8 yo',  '8 yr',   '9 yo',  '9 yr',   '10 yo',  '10 yr',   
                  '11 yo',  '11 yr',   '12 yo',  '12 yr',   '13 yo',  '13 yr',   '14 yo',  '14 yr',   '15 yo',  '15 yr',   '16 yo',  '16 yr',   '17 yo',  '17 yr',   
                  'yo',  'yr',  }
  }
