from datetime import datetime

def create_site_log(fn, sitename, year):
    f = open(fn, 'w')
    f.write(f'POST PROCESSING LOG FILE FOR: {sitename}\n')
    f.write(f'DATA COLLECTION YEAR: {year}\n')
    f.write(f'LOG CREATED: {datetime.now().strftime("%m/%d/%Y")}\n')
    f.write('\n\n')
    f.close()
    
def append_to_log(fn, header, content):
    f = open(fn, 'a')
    f.write('\n')
    f.write(header.upper() + '\n')
    f.write('\n')
    f.write(content)
    f.write('\n')
    f.close()     