
def yes_or_no(question):
    while True:
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply == 'y':
            return True
        elif reply == 'n':
            return False
        else:
            print('your answer is invalid. Please re-enter')

def number_input(question):
    while True:
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply.isnumeric():
            return True
        else:
            print('your answer is invalid. Please re-enter')
