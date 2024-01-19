from garnet.models import data_catalog
from garnet.config import instruments

login = data_catalog.oncat_login()

def test_topaz():

    ipts_number = 31189
    title = 'YAG'
    instrument = instruments.topaz
    
    data_files = data_catalog.retrieve_data_files(login,
                                                  instrument, 
                                                  ipts_number)
        
    run_title_dict = data_catalog.run_title_dictionary(data_files, instrument)
    
    rs = run_title_dict[title]

    run_list = data_catalog.run_numbers_list(rs)

    goniometer_entries_list = data_catalog.goniometer_entries(instrument)

    for goniometer_entry in goniometer_entries_list:
        print([df[goniometer_entry] for df in data_files])

test_topaz()