import pandas as pd
import numpy as np
import re
from collections import Counter
from datetime import datetime

# First step of cleaning
def remove_delim(text):
    ''' 
    Replaces delimeters from a Pandas Series with a space.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): Series with replaced delimeters
    '''
    return text.replace(r'\n|\*|--+|\s-|-\s|/'," ", regex = True)

def remove_email(text):
    ''' 
    Removes emails of the xxx@xxx.xxx or https:\\xxx@xxx.xxx from a Pandas Series.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): Series with removed emails
    '''
    text = text.replace(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)',"", regex = True)
    return text.replace(r"\S*@\S*\s?", " ", regex = True)

def remove_slashes(text):
    ''' 
    Replaces "\\" expressions from a Pandas Series with a space.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): Series with replaced "\\" expressions.
    '''
    return text.replace(r"\\"," ")

def remove_links(text):
    ''' 
    Replaces links of the www.xxx, WWW.xxx and http:xxxx and https:xxx expressions from a Pandas Series with a space.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): Series with replaced link expressions.
    '''
    text = text.replace(r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?'," ", regex = True)
    text = text.replace(r'\s*?www\.\S*\.[A-Za-z]{2,5}\s*'," ", regex = True)
    text = text.replace(r'\s*?WWW\.\S*\.[A-Za-z]{2,5}\s*'," ", regex = True)
    return text

def mod_remove_links(text):
    ''' 
    Replaces links of the www.xxx, WWW.xxx and http:xxxx and https:xxx expressions from a Pandas Series with a space.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): string with replaced link expressions.
    '''
    text = text.replace(r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?'," ")
    text = text.replace(r'\s*?www\.\S*\.[A-Za-z]{2,5}\s*'," ")
    text = text.replace(r'\s*?WWW\.\S*\.[A-Za-z]{2,5}\s*'," ")
    return text

def remove_bullet(text):
    ''' 
    Replaces bullet points with spaces.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): Series with replaced bullet points.
    '''
    return text.replace(r'•|●'," ", regex = True)

def remove_image(text):
    ''' 
    Replaces images in formats commonly seen in the source data, enclosed in square brackets.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): Series with replaced image formats.
    '''
    text = text.replace("[Spacer]", " ")
    text = text.replace("[Twitter]", " ")
    text = text.replace("[Linkedin]", " ")
    text = text.replace("[Instagram]", " ")
    text = text.replace("[Email Signature]", " ")
    text = text.replace("[logo]", " ")
    text = text.replace("[Logo]", " ")
    text = text.replace("[Image removed by sender. Logo]", " ")
    text = text.replace("[Email Logo Template]", " ")
    text = text.replace("[image]", " ")
    text = text.replace("[]", " ")
    return text

def remove_deskdic_question(text):
    ''' 
    Removes the deskdirector questions, which appear in every entry from this source.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): Series with replaced deskdirector questions.
    '''
    text = text.replace(r"summary of issue", " ", regex = True)
    text = text.replace(r"details of issue", " ", regex = True)
    text = text.replace(r"if your callback number is different than what's on record, please provide it below", " ", regex = True)
    text = text.replace(r"have you opened a ticket about this issue before", " ", regex = True)
    text = text.replace(r"how many users are impacted by this issue", " ", regex = True)
    text = text.replace(r"how would you classify this issue", " ", regex = True)
    text = text.replace(r"what company is this quote for", " ", regex = True)
    text = text.replace(r"who should the quote be addressed to", " ", regex = True)
    text = text.replace(r"who should it be addressed to", " ", regex = True)
    text = text.replace(r"which location is the product for", " ", regex = True)
    text = text.replace(r"which nd location is it needed at", " ", regex = True)
    text = text.replace(r"when is it needed by or when is the next site visit for the client's location", " ", regex = True)
    text = text.replace(r"what do you need quoted", " ", regex = True)
    return text

def remove_greeting(text):
    ''' 
    Removes greetings from the text.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): Series with replaced deskdirector questions.
    '''
    text = text.replace(r"good morning", " ", regex = True)
    text = text.replace(r"good afternoon", " ", regex = True)
    text = text.replace(r"good evening", " ", regex = True)
    text = text.replace(r"good day", " ", regex = True)
    return text

def remove_punctuation(text):
    ''' 
    Removes punctuation except "!","'","?",",",".",":" and round brackets.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): Series with removed puncutation.
    '''
    return text.replace(r'[^\w\!\'\?\,\.\:\)\(]+', " ", regex = True)

def remove_num_longer_than_3(text):
    ''' 
    Removes numbers longer than 3 digits.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): Series with removed long numbers.
    '''
    return text.replace(r'\d{3,}'," ", regex = True)

def remove_whitespace(text):
    ''' 
    Removes white spaces and consecutive dots.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): Series with removed whitespaces and consecutive dots
    '''
    text = text.replace(r' (\- )+ | (\.)+ | (\- )+ | (\+ )+ | ( ) '," ")
    text = text.replace(r' . . ',"")
    text = text.replace("..","")
    text = text.replace(". .",".")
    return text.replace(r'\s{2,}'," ", regex = True)

def remove_email_items(text):
    ''' 
    Removes common email signatures and other objects.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): Series with removed email items
    '''
    text = text.replace('caution:this email originated from outside of the mnp network. be cautious of any embedded links and/or attachments.','')
    text = text.replace('mise en garde:ce courriel ne provient pas du réseau de mnp. méfiez-vous des liens ou pièces jointes qu’il pourrait contenir.','')
    text = text.replace('caution: this email originated from outside of the mnp network. be cautious of any embedded links and/or attachments.','')
    text = text.replace('mise en garde: ce courriel ne provient pas du réseau de mnp. méfiez-vous des liens ou pièces jointes qu’il pourrait contenir.','')
    text = text.replace(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",'')
    text = text.replace(r"^(from:).*",'')
    text = text.replace(r"^(sent:).*",'')
    text = text.replace(r"^(to:).*",'')
    text = text.replace(r"^(cc:).*",'')
    text = text.replace(r"^(bcc:).*",'')
    text = text.replace(r"^(subject:).*",'')
    text = text.replace(r"^(importance:).*",'')
    text = text.replace(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",'')
    text = text.replace(r"(\[(.*?)\])",'')
    text = text.replace(r"(\d)\D*(\d{3})\D*(\d{3})\D*(\d{4})", '')
    text = text.replace("(\d{3})\D*(\d{3})\D*(\d{4})",'')
    return text

def email_splitter(text):
    ''' 
    Divides emails into two parts after detecting words that commonly signify the end of the message, such as thanks, regards and sincerely, as well as other common words seen in signatures, like confidential.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): The first part of the division
    '''
    text = text.str.lower().str.split(r"(thank).*", n = 1, expand = True)[0]
    text = text.str.lower().str.split(r"(regards).*", n = 1, expand = True)[0]
    text = text.str.lower().str.split(r"(sincerely).*", n = 1, expand = True)[0]
    text = text.str.lower().str.split(r"(cheers).*", n = 1, expand = True)[0]
    text = text.str.lower().str.split(r"(confidential).*", n = 1, expand = True)[0]
    # text = text.str.lower().str.split(r"(caution).*", n = 1, expand = True)[0]
    text = text.str.lower().str.split(r"(the information contained in this).*", n = 1, expand = True)[0]
    text = text.str.lower().str.split(r"(this email and any files transmitted).*", n = 1, expand = True)[0]
    text = text.str.lower().str.split(r"(this email originated).*", n = 1, expand = True)[0]
    text = text.str.lower().str.split(r"(this message is).*", n = 1, expand = True)[0]
    return text

def clean_text(text):
    ''' 
    Ensures only ASCII encoding passes and applies an ensemble of previous functions in succesion, and repetition, to eliminate them.
    ---
    Parameters:
        text (series): A Pandas Series holding strings.
    ---
    Returns:
        text (series): ASCII text with cleaning functions applied.
    '''
    text = text.str.encode('ascii', 'ignore').str.decode('ascii')
    text = remove_email_items(text)
    text = remove_delim(text)
    text = remove_email(text)
    text = remove_links(text)
    text = remove_bullet(text)
    text = remove_image(text)
    text = remove_num_longer_than_3(text)
    # text = remove_punctuation(text)
    text = remove_whitespace(text)
    text = remove_deskdic_question(text)
    text = remove_greeting(text)
    return text

# Email signature removal functions
def mod_remove_punctuation(text):
    ''' 
    Removes punctuation except "!","'" and round brackets.
    ---
    Parameters:
        text (string): .
    ---
    Returns:
        text (string): Series with removed puncutation.
    '''
    return re.sub(r'[^\w\!\']+', " ", text, flags=re.MULTILINE)

def mod_remove_email(text):
    ''' 
    Removes emails of the xxx@xxx.xxx from a string.
    ---
    Parameters:
        text (series): A string.
    ---
    Returns:
        text (series): String with removed emails
    '''
    return text.replace(r"\S*@\S*\s?", "")

def mod_remove_bullet(text):
    ''' 
    Replaces bullet points with spaces.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): Series with replaced bullet points.
    '''
    return text.replace(r'•|●'," ")

def mod_remove_num_longer_than_3(text):
    return text.replace(r'\d{3,}'," ")

def mod_remove_image(text):
    ''' 
    Replaces images in formats commonly seen in the source data, enclosed in square brackets.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): a string without image references.
    '''
    text = text.replace("[Spacer]", " ")
    text = text.replace("[Twitter]", " ")
    text = text.replace("[Linkedin]", " ")
    text = text.replace("[Instagram]", " ")
    text = text.replace("[Email Signature]", " ")
    text = text.replace("[logo]", " ")
    text = text.replace("[Logo]", " ")
    text = text.replace("[Image removed by sender. Logo]", " ")
    text = text.replace("[Email Logo Template]", " ")
    text = text.replace("[image]", " ")
    return text

def mod_remove_delim(text):
    ''' 
    Replaces delimeters from a string.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): String with replaced delimeters
    '''
    return re.sub(r'\*|--+|\s-|-\s|/'," ",text,flags=re.MULTILINE)

def mod_clean_text(text):
    ''' 
    Applies an ensemble of cleaning functions to a string.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): a string without delimeters, emails, links, bullet points, images or numbers longer than 3 digits.
    '''
    text = mod_remove_delim(text)
    text = mod_remove_email(text)
    text = mod_remove_links(text)
    text = mod_remove_bullet(text)
    text = mod_remove_image(text)
    text = mod_remove_num_longer_than_3(text)
    return text

def check_name(text,ticket_name=[]):
    ''' 
    Checks whether ticket_name is in text.
    ---
    Parameters:
        text (string): a string.
        ticket_name (list): a list containing the name you want to look for in the string
    ---
    Returns:
        text (series): a boolean on whether text contains the name.
    '''
    text = text.lower()
    text = text.split(",")
    text = text[0].strip()
    return text in ticket_name

def check_greeting(text,stop_saying=[],stop_list=[]):
    ''' 
    Checks whether stop_saying or stop_list is in text, used to check if a greeting is in a specific line.
    ---
    Parameters:
        text (string): a string.
        stop_saying (list): a list containing sentences that are to be searched for in the text
        stop_list (list): a list of word to be searched for in the text
    ---
    Returns:
        text (series): a boolean on whether text contains the name.
    '''
    text = text.strip()
    if "|" in text:
        return True
    # text = mod_remove_punctuation(text)
    if text == "":
        return True
    text = text.lower()
    if text in stop_saying:
        return True
    text = text.split(" ")
    return text[0] in stop_list

def check_flag_words(text,stop_saying=[],stop_list=[]):
    ''' 
    Checks whether stop_saying or stop_list is in text, used to check if a common sentence is in a specific line.
    ---
    Parameters:
        text (string): a string.
        stop_saying (list): a list containing sentences that are to be searched for in the text
        stop_list (list): a list of word to be searched for in the text
    ---
    Returns:
        text (series): a boolean on whether text contains the name.
    '''
    text = text.strip()
    if "|" in text:
        return True
    # text = mod_remove_punctuation(text)
    if text == "":
        return True
    text = text.lower()
    if text in stop_saying:
        return True
    text = text.split(" ")
    return text[0] in stop_list

def email_divisions(text, contact_name):
    ''' 
    Checks text line by line for lines that can be classified as greetings or signatures and brings only the lines that are not. This is useful to get only the relevant lines on the ticket.
    ---
    Parameters:
        text (string): a string.
        contact_name (string): the name of the person who submitted the ticket. Is used to remove said name and any line containing it.
    ---
    Returns:
        Series: a Pandas series with only relevant text.
    '''
    alphabet_list = [letter for letter in list(map(chr, range(97, 123))) if letter not in ["a","i"]]
    greeting_saying = ["good morning","have a great day","good day","good afternoon","good evening,","good morning,","have a great day,","good day,","good afternoon,","good evening,","hello,","hi,","Good morning."]
    greeting_list=["hi","hello","morning","hey"]
    flag_saying=["if you have any questions or concerns please let me know.","let me know","this message is intended",""]+greeting_saying
    flag_list=["to","from","w","to","from","cc","http","ph","sent","ok","ext"]+greeting_list+alphabet_list
    name_list = [str(name).lower() for name in contact_name.values]
    parsed_string = []
    count = 0
    for ticket in range(0,len(text)):
        holder = ""
        for line in [re.split("\n|\r",block) for block in re.split("\n\n|\r\n",mod_clean_text(text[ticket].lower()))]:
          holder += " "
          if not(check_name(line[0],name_list)) and not(check_flag_words(line[0],flag_saying,flag_list)):
              holder += " ".join(line)
          elif check_greeting(line[0],greeting_saying,greeting_list): #The starting word is Hi or some greeting
              intermediate = re.split("\,|\.|\!|\n| ^",line[0])
              if len(intermediate) <= 2:
                  if len(line) > 1 and not check_flag_words(line[1],flag_saying,flag_list):
                      holder += " ".join(line[1:])
              elif not check_flag_words(intermediate[1],flag_saying,flag_list):
                  holder += " ".join(intermediate[1:])
        parsed_string.append(holder)
    return pd.Series(parsed_string)

# Combined summary + description clean-up functions
def remove_mnp_caution(text):
    ''' 
    Removes lines related to email signatures.
    ---
    Parameters:
        text (Series): a Pandas string series.
    ---
    Returns:
        text (series): a Panda text series without these signatures.
    '''
    text = text.replace("caution: this email originated from outside of the mnp network.", " ")
    text = text.replace("be cautious of any embedded links and or attachments.", " ")
    text = text.replace(" mise en garde:ce courriel provient pas du rseau de mnp.", " ")
    text = text.replace("caution:this email originated from outside of the mnp network.", " ")
    text = text.replace(" mise en garde: ce courriel provient pas du rseau de mnp.", " ")
    text = text.replace("mfiez-vous des liens ou pices jointes quil pourrait contenir.", " ")
    return text

def remove_windows_high_severity_alert(text):
    ''' 
    Replaces a common automated alert from Windows Compliance Center.
    ---
    Parameters:
        text (Series): a Pandas string series.
    ---
    Returns:
        text (series): a Panda text series replacing all instances of this message
    '''
    if ("severity alert: Microsoft compliance center." in text):
        return "Microsoft compliance centre."
    else:
        return text
    
def remove_mhk_footer(text):
    ''' 
    Replaces a common signature from MHK clients.
    ---
    Parameters:
        text (Series): a Pandas string series.
    ---
    Returns:
        text (series): a Panda text series replacing all instances of this message
    '''
    return re.sub(r"we're here to help with your insurance needs emails and phone calls are still encouraged appointments are required for in office broker meetings please wear a mask when visiting mhk welcomes e transfer payments to if you receive this email in error please notify us by reply email and destroy this message mhk complies with canada's anti spam and alberta's pipa legislations if you no longer wish to receive emails from mhk please reply with 'unsubscribe' in the subject line we're here to help with your insurance needs emails and phone calls are still encouraged appointments are required for in office broker meetings please wear a mask when visiting mhk welcomes e transfer payments to if you receive this email in error please notify us by reply email and destroy this message mhk complies with canada's anti spam and alberta's pipa legislations if you no longer wish to receive emails from mhk please reply with 'unsubscribe' in the subject line"\
                 ," ", text)

def remove_carya_footer(text):
    ''' 
    Replaces a common signature from Carya clients.
    ---
    Parameters:
        text (Series): a Pandas string series.
    ---
    Returns:
        text (series): a Panda text series replacing all instances of this message
    '''
    text = text.replace("carya (formerly family services) stay up to date with the latest carya news, programs, and events by signing up for ourmonthly newsletter.","")
    text = text.replace("in the spirit of our efforts to promote reconciliation, we acknowledge the traditional territories and oral practices of the blackfoot, the tsuut'ina, the stoney nakoda first nations, the mtis nation region , and all people who make their homes in the treaty region of southern we also respectfully acknowledge that the province of is comprised of treaty , treaty , and treaty territory, the traditional lands of first nations and mtis peoples.","")
    text = text.replace("no form of electronic communication is secure and may be intercepted by others.","")
    text = text.replace("carya cannot guarantee the receipt of electronic communication nor a timely response. where communication is.","")
    return text

def remove_capital_paper_footer(text):
    ''' 
    Replaces a common signature from Capital paper clients.
    ---
    Parameters:
        text (Series): a Pandas string series.
    ---
    Returns:
        text (series): a Panda text series replacing all instances of this message
    '''
    return re.sub(r"leaders in paper recovery effective immediately due to the unsecured nature we cannot accept interac e transfers unless authorized by kim burns the information in this email and any attachments is sent by capital paper recycling ltd and is intended to be confidential and for the use of only the individual or entity named above the information may be protected by solicitor client privilege work product immunity or other legal principles if the reader of this message is not the intended recipient you are notified that unauthorized review retention dissemination distribution copying or other use of or taking any action in reliance upon this information is strictly prohibited if you received this email in error please notify us immediately by email reply and delete or destroy this message and any copies "\
                 , " ", text)

def remove_postalcode(text):
    ''' 
    Removes all instances of xxx xxx postal codes.
    ---
    Parameters:
        text (Series): a Pandas string series.
    ---
    Returns:
        text (series): a Panda text without postal codes in this form
    '''
    return re.sub(r'[A-Za-z]{1}[0-9]{1}[A-Za-z]{1}(\s*|-)[0-9]{1}[A-Za-z]{1}[0-9]{1}|[A-Za-z]{1} [0-9]{1} [A-Za-z]{1} [0-9]{1} [A-Za-z]{1} [0-9]{1}',"",text)

def remove_jira(text):
    ''' 
    Replaces a common signature from Jira clients.
    ---
    Parameters:
        text (Series): a Pandas string series.
    ---
    Returns:
        text (series): a Panda text series replacing all instances of postal codes.
    '''
    text = text.replace("view issue get jira notifications on your phone!","")
    text = text.replace("download the jira cloud app for android or ios.","")
    text = text.replace("manage notifications give feedback privacy policy.","")
    return text

def annoying_things(text):
    ''' 
    Replaces common repeated objects in the emails such as telephones, cellphones and random round brackets or triple apostrophes.
    ---
    Parameters:
        text (Series): a Pandas string series.
    ---
    Returns:
        text (series): a Panda text series replacing all instances of this these objects.
    '''
    text = text.replace("|", " ")
    text = text.replace("tel:", " ")
    text = text.replace("cell:"," ")
    text = text.replace("( )"," ")
    text = text.replace("()"," ")
    text = text.replace("```", " ")
    text = text.replace("\t"," ")
    text = text.replace(": :"," ")
    return text

def remove_numbers(text):
    ''' 
    Replaces numbers with empty space.
    ---
    Parameters:
        text (Series): a Pandas string series.
    ---
    Returns:
        text (series): a Panda text series replacing all numbers.
    '''
    return re.sub(r'\d'," ",text)

def remove_hashtags(text):
    ''' 
    Replaces triple hashtags caused by DeskDic.
    ---
    Parameters:
        text (Series): a Pandas string series.
    ---
    Returns:
        text (series): a Panda text series replacing all triple hashtags.
    '''
    return text.replace("###","")

def replace_locations(text):
    ''' 
    Replaces all common abbreviated locations present in signatures like alberta, edmonton, ab, sw, nw, etc.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): a string replacing all instances of these messages.
    '''
    text = re.sub(r"alberta"," ",text)
    text = re.sub(r"edmonton"," ",text)
    text = re.sub(r"calgary"," ",text)
    text = re.sub(r" ab "," ",text)
    text = re.sub(r" sw ", " ", text)
    text = re.sub(r" nw "," ",text)
    text = re.sub(r" ne ", " ", text)
    text = re.sub(r" se "," ",text)
    text = re.sub(r" ca "," ",text)
    text = re.sub(r" ave "," ",text)
    return text

def remove_email_row(text):
    ''' 
    Replaces instances of emails of the form xxx@xxx.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): a string replacing all xxx@xxx emails.
    '''
    text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)',"",text)
    return re.sub(r'\S*@\S*\s?'," ",text)

def remove_brackets(text):
    ''' 
    Replaces left over brackets.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): a string replacing all brackets.
    '''
    return re.sub(r'\[[^)]*\]',"",text)

def remove_whitespace_row(text):
    ''' 
    Replaces whitespaces, consecutive dots and other left overs from previous steps on a string.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): a string without whitespaces, double dots and other anomalies created by previous steps.
    '''
    text = re.sub(r' (\- )+ | (\.)+ | (\- )+ | (\+ )+ | ( ) '," ",text)
    text = re.sub(r' . . ',"",text)
    text = text.replace("..","")
    text = text.replace(". .",".")
    text = re.sub(r'\s{2,}'," ",text)
    return text.replace(" .","")

def remove_renewal_end(text):
    ''' 
    Replaces common renewal messages with the word renewal for ease of modelling.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): a string replacing all such instances.
    '''
    text = re.sub(r' -[A-z][A-z][A-z]- monitoring indicates that there is a configuration expiring in the next days. see attached configuration for details|monitoring indicates that there is a configuration expiring in the next days. see attached configuration for details',"",text)
    if text == "":
        text = "renewal"
    return text

def remove_escalation_end(text):
    ''' 
    Replaces common escalation messages with the word renewal for ease of modelling.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): a string replacing all such instances.
    '''
    return re.sub(r', config attached, which expires within days',".",text)

def extra_cleaning(text):
    ''' 
    Removes and replaces common themes in the dataset that prevent accurate modelling by cluttering with irrelevant information.
    ---
    Parameters:
        text (string): a string.
    ---
    Returns:
        text (string): a string replacing these problematic messages.
    '''
    text = remove_postalcode(text)
    text = replace_locations(text)
    text = remove_hashtags(text)
    text = remove_brackets(text)
    text = remove_email_row(text)
    text = remove_numbers(text)
    text = annoying_things(text)
    text = remove_whitespace_row(text)
    text = remove_carya_footer(text)
    text = remove_windows_high_severity_alert(text)
    text = remove_mhk_footer(text)
    text = remove_capital_paper_footer(text)
    text = remove_jira(text)
    text = remove_mnp_caution(text)
    text = remove_renewal_end(text)
    text = remove_escalation_end(text)
    text = remove_whitespace_row(text)
    return text

def description_cleaner(description, contact_name):
    ''' 
    Removes and replaces common themes in the descriptions that prevent accurate modelling by cluttering with irrelevant information.
    ---
    Parameters:
        text (series): a Pandas Series containing strings.
        contact_name (string): the name of the person who submitted the ticket. Is used to remove said name and any line containing it.
    ---
    Returns:
        clean_description (series): a Pandas series replacing irrelevant objects and problems.
    '''
    clean_description = email_divisions(description, contact_name)
    clean_description = remove_deskdic_question(clean_description)
    clean_description = clean_description.str.lower()
    clean_description = clean_text(clean_description)
    clean_description = email_splitter(clean_description)
    return clean_description

def summary_cleaner(summary, contact_name, company_name):
    ''' 
    Removes and replaces common themes in the summaries that prevent accurate modelling by cluttering with irrelevant information.
    ---
    Parameters:
        text (series): a Pandas Series containing strings
        contact_name (string): the name of the person who submitted the ticket. Is used to remove said name and any line containing it.
        company_name (string): the name of the person who submitted the ticket. Is used to remove said name and any line containing it.
    ---
    Returns:
        clean_summary (series): a Pandas series replacing irrelevant objects and problems.
    '''
    clean_summary = summary.str.lower()#.str.split(r"-", n = 1, expand = True)[1]
    for line in range(len(summary)):
      clean_summary[line] = str(clean_summary[line]).replace("nan", "")
      clean_summary[line] = clean_summary[line].encode('ascii', 'ignore').decode('ascii')
      clean_summary[line] = str(clean_summary[line]).replace(str(contact_name[line]), "")
      clean_summary[line] = str(clean_summary[line]).replace(str(company_name[line]), "")
      if "-" in str(clean_summary[line]):
        clean_summary[line] = summary[line].split(r"-", 1)[1] if (summary[line].split(r"-", 1)[1] != "") else summary[line].split(r"-")[0]
      clean_summary[line] = clean_summary[line].strip() if (clean_summary[line].strip() != "") else clean_summary[line].strip()
    return clean_summary

def summary_description_combination(summary, description, contact_name, company_name):
    ''' 
    Removes and replaces common irrelevant themes in the descriptions and summaries while also combining them into a single line.
    ---
    Parameters:
        summary (series): a Pandas Series containing strings
        description (series): a Pandas Series containing strings
        contact_name (string): the name of the person who submitted the ticket. Is used to remove said name and any line containing it.
        company_name (string): the name of the person who submitted the ticket. Is used to remove said name and any line containing it.
    ---
    Returns:
        combined_text (series): a Pandas series combining description and summary.
    '''
    summary = summary.astype(str)
    description = description.astype(str)
    contact_name = contact_name.astype(str)
    company_name = company_name.astype(str)
    clean_summary = summary_cleaner(summary, contact_name, company_name)
    clean_description = description_cleaner(description, contact_name)
    combined_text = clean_summary + ". " + clean_description + "."
    for i in range(0,len(combined_text)):
        combined_text[i] = extra_cleaning(combined_text[i])
    combined_text.name = "combined_text"
    return combined_text

def extra_features(clean_dataset):
    ''' 
    Adds one-hot-encoded features to a dataset to facilitate modelling.
    ---
    Parameters:
        clean_dataset (dataframe): a Pandas Series containing strings
    ---
    Returns:
        clean_dataset (dataframe): a dataframe including additional features.
    '''
    ### One-Hot-Encode Source 
    email_connector = [0]*len(clean_dataset)
    deskdirector = [0]*len(clean_dataset)
    email = [0]*len(clean_dataset)
    renewal = [0]*len(clean_dataset)
    escalation = [0]*len(clean_dataset)
    for i in range(0,len(clean_dataset.Source)):
        if clean_dataset.Source[i] == "Email Connector":
            email_connector[i] = 1
        elif clean_dataset.Source[i] == "DeskDirector":
            deskdirector[i] = 1
        elif clean_dataset.Source[i] == "Email":
            email[i] = 1
        elif clean_dataset.Source[i] == "Renewal":
            renewal[i] = 1
        elif clean_dataset.Source[i] == "Escalation":
            escalation[i] = 1 
    clean_dataset = clean_dataset.assign(email_connector = email_connector, deskdirector=deskdirector, email=email, renewal=renewal, escalation=escalation)

    ### One-Hot-Encode Board
    brd36 = [0]*len(clean_dataset)
    brd41 = [0]*len(clean_dataset)
    brd43 = [0]*len(clean_dataset)
    for i in range(0,len(clean_dataset)):
        if clean_dataset.Board[i] == 36:
            brd36[i] = 1
        elif clean_dataset.Board[i] == 41:
            brd41[i] = 1
        elif clean_dataset.Board[i] == 43:
            brd43[i] = 1
    clean_dataset["brd36"] = brd36
    clean_dataset["brd41"] = brd41
    clean_dataset["brd43"] = brd43

    #1 vs 2/3 Split and 1/2 vs 3 Split 
    low_medhigh = [0]*len(clean_dataset)
    lowmed_high=[0]*len(clean_dataset)
    for i in range(0,len(clean_dataset.Source)):
        if clean_dataset.Severity[i] >= 1:
            low_medhigh[i] = 1
            if clean_dataset.Severity[i] == 2:
                lowmed_high[i] = 1
    clean_dataset["Low_vs_MedHigh"] = low_medhigh
    clean_dataset["LowMed_vs_High"] = lowmed_high
    return clean_dataset

def clean_tickets(ticketNbr, contact_name, company_name, Summary, Initial_Description, Impact, Severity, Board, Source, date_entered):
    ''' 
    Wrapper of cleaning functions that removes problematic and irrelevant elements from said dataset and one-hot encodes additional features for ease of modelling.
    Inputs required are each of the columns in the format extracted from the dataset.
    ---
    Parameters:
        ticketNbr = Pandas Series containing the Ticket Numbers from the ticketing system
        contact_name = Pandas Series containing the contact names of the users submitting the tickets
        company_name = Pandas Series containing the company names from the ticket's sources
        Summary = Pandas Series containing the text summaries/email titles.
        Initial_Description = Pandas Series containing the bulk of the message or, in DeskDirector cases, answers to their form.
        Impact = Pandas Series containing the ticket's Impact. 1 to 3.
        Severity = Pandas Series containing the ticket's Severity. 1 to 3.
        Board = Pandas Series containing the ticket's Board, three categories.
        Source = Pandas Series detailing the source of the ticket. Two main categories are more relevant: DeskDirector and Email Connector.
        date_entered = Pandas Series containing the date the ticket was submitted in.
    ---
    Returns:
        clean_dataset (dataframe): a dataframe containing additional features and with common irrelevant themes and objects removed or replaced.
    '''
    combined_text = summary_description_combination(Summary, Initial_Description, contact_name, company_name)
    clean_dataset = pd.concat([combined_text, Impact.rename("Impact"), Severity.rename("Severity"), Board.rename("Board"), Source.rename("Source")], axis=1)
    clean_dataset = extra_features(clean_dataset)
    clean_dataset.Severity = clean_dataset.Severity -1
    clean_dataset.Impact = clean_dataset.Impact -1
    return clean_dataset