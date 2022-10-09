def get_lang_groups(src_lang):
    """ we use langid because it's pretty fast but it has difficulties in low resource languages
    langid can sometimes mistake languages that are in the same group. that is ok for our purpose as
    we mainly use the langid check to confirm the labels from other models. """
    lang_groups=[src_lang]
    if src_lang in ('ig', 'sn', 'ny', 'st', 'zu', 'xh', 'rw', 'sw', 'yo', 'so'):
      lang_groups = ['ig', 'sn', 'ny', 'st', 'zu', 'xh', 'rw', 'sw', 'yo', 'so']
    elif src_lang in ('mr', 'ne', 'hi', ):
      lang_groups = ['mr', 'ne', 'hi', ]
    elif src_lang in ('fr', 'br'):
      lang_groups = ['fr','la', 'br' ]
    elif src_lang in ('pt', ):
      lang_groups = ['pt','la', 'gl' ]
    elif src_lang in ('eo', 'es', 'oc', 'ca', 'eu', 'an', 'gl' ):
      lang_groups = ['eo', 'es', 'oc', 'ca', 'eu', 'an', 'gl', 'la' ]
    elif src_lang in ('arz', 'ar', 'fa', 'ur', 'az', 'azb', 'ckb', 'ps' ):
      lang_groups = ['arz', 'ar', 'fa', 'ur', 'az', 'azb', 'ckb', 'ps' ]
    elif src_lang in ('id', 'ms', ):
      lang_groups = ['id', 'ms',]
    elif src_lang in ('as', 'bn', 'bpy'):
      lang_groups = ['as', 'bn', 'bpy']
    elif src_lang in ('af', 'nl', ):
      lang_groups = ['af', 'nl',]
    elif src_lang in ('bo', 'dz', ):
      lang_groups = ['bo', 'dz',]
    elif src_lang in ('bs', 'hr', ):
      lang_groups = ['bs', 'hr',]
    elif src_lang in ('bxr', 'mn', ):
      lang_groups = ['bxr', 'mn',]
    elif src_lang in ('ceb', 'tl', ):
      lang_groups = ['ceb', 'tl',]
    elif src_lang in ('cs', 'sk', ):
      lang_groups = ['cs', 'sk',]
    elif src_lang in ('da', 'no', ):
      lang_groups = ['da', 'no',]
    elif src_lang in ('eml', 'wa', ):
      lang_groups = ['eml', 'wa',]
    elif src_lang in ('de', 'lb', 'pl', 'dsb'):
      lang_groups = ['de', 'lb', 'pl', 'dsb']
    elif src_lang in ('id', 'jv', 'ms', 'tl',):
      lang_groups = ['id', 'jv', 'ms', 'tl', ]
    elif src_lang in ('av', 'ru', 'bg', 'ba', 'kk', 'ky', 'uk', 'be', 'ce', 'cv'):
      lang_groups = ['av', 'ru', 'bg', 'ba', 'kk', 'ky', 'uk', 'be', 'ce', 'cv']
    return set(lang_groups)
