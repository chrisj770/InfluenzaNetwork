import csv
import sys

if __name__=="__main__": 
    '''
    Retrieve filenames from command line
    Usage: datasort.py <path_to_influenza_file> <path_to_census_file> <path_to_svi_file1>...
    '''
    influenzaFilename = None
    populationFilename = None
    sviFilenames = []
    try:
        influenzaFilename = str(sys.argv[1])
        populationFilename = str(sys.argv[2])
        if (not ".csv" in influenzaFilename) or (influenzaFilename is None): 
            raise Exception()
        if (not ".csv" in populationFilename) or (populationFilename is None): 
            raise Exception()
        for sviFilename in sys.argv[3:]:
            if (".csv" in sviFilename):
                sviFilenames.append(sviFilename)
    except Exception:
        print("Error: No file specified!")
        print("Usage: datasort.py <path_to_influenza_file> <path_to_census_file> <path_to_svi_file1>...")
        sys.exit(0)
        
    '''
    Load Influenza CSV file into memory
    Expect Fields: Season, Region, County, CDC Week, Week Ending Date, Disease, Count, County Centroid, FIPS
    '''
    yearSet = {}
    with open(influenzaFilename, 'r') as rp: 
        csvreader = csv.reader(rp)
        fields = next(csvreader)        
        # Grab individual years, counties, diseases           
        for row in csvreader: 
            year = row[0]
            county = row[2] 
            diseaseCount = int(row[6])
            if year not in yearSet: 
                yearSet[year] = {county:[diseaseCount]}
            else: 
                if county not in yearSet[year]: 
                    yearSet[year][county] = [diseaseCount]
                else: 
                    yearSet[year][county][0] += diseaseCount
        
    '''
    Load Census CSV file into memory 
    Expect Fields: ... , STNAME, CTYNAME, ... , POPESTIMATE2010, POPESTIMATE2011, ... , POPESTIMATE2019, ...
    '''
    with open(populationFilename, 'r') as rp: 
        csvreader = csv.reader(rp)
        fields = next(csvreader)        
        # Grab individual state names, county names, population estimates         
        for row in csvreader: 
            STNAME = row[5]
            CTYNAME = row[6]
            POPESTIMATE = []
            for i in range(9, 19):
                POPESTIMATE.append(row[i])                 
            # Determine if data for a County in NY
            if STNAME == "New York": 
                if "County" in CTYNAME:
                    countyName = CTYNAME[:CTYNAME.find("County")].strip().upper()                    
                    # Add each year's data to county's entry in year dictionary
                    for year in yearSet: 
                        targetYear = year[year.find("-")+1:]
                        targetCell = int(targetYear)-2010
                        if (countyName in yearSet[year]) and (targetCell < len(POPESTIMATE)): 
                            population = int(POPESTIMATE[targetCell])
                            percentage = float(yearSet[year][countyName][0]) / float(population)
                            yearSet[year][countyName].append(population)
                            yearSet[year][countyName].append(percentage)
    '''
    Load Social Vulnerability Index (SVI) file into memory 
    Expect Fields as described in dictionary "fieldSort" (See below...)
    '''                    
    yearSet1 = {}
    fieldSort = {"STATE": None, "COUNTY" : None, "EP_POV" : None, "EP_UNEMP" : None, "EP_PCI" : None, "EP_NOHSDP" : None, "EP_AGE65" : None, "EP_AGE17" : None, "EP_DISABL" : None, "EP_SNGPNT" : None, "EP_MINRTY" : None, "EP_LIMENG" : None, "EP_MUNIT" : None, "EP_MOBILE" : None, "EP_CROWD" : None, "EP_NOVEH" : None, "EP_GROUPQ" : None, "EP_UNINSUR" : None}
    for filename in sviFilenames:
        fileyear = int(filename[filename.find("svi")+3:filename.find(".csv")])
        if fileyear not in yearSet1: 
            yearSet1[fileyear] = {}
        with open(filename, 'r') as rp: 
            csvreader = csv.reader(rp)
            fields = next(csvreader) 
            for field in range(len(fields)):        
                if fields[field] == "STATE": 
                    fieldSort["STATE"] = field
                elif fields[field] == "COUNTY": 
                    fieldSort["COUNTY"] = field
                else: 
                    fieldSplit = fields[field][:fields[field].find("_")].strip()
                    if fieldSplit == "EP":
                        fieldSort[fields[field]] = field
            for row in csvreader: 
                if row[fieldSort["STATE"]] == "NEW YORK": 
                    county = row[fieldSort["COUNTY"]].upper()
                    if not (county in yearSet1):
                        yearSet1[fileyear][county] = {}
                    for key in sorted(fieldSort.keys()):
                        if key in ["STATE", "COUNTY"]:
                            continue
                        if fieldSort[key] is None: 
                            print("CANNOT FIND ONE")
                            yearSet1[fileyear][county][key] = -999999999;
                        else:
                            yearSet1[fileyear][county][key] = row[fieldSort[key]]
    '''
    Write All fields to CSV
    '''
    sortedyears = list(yearSet.keys())
    sortedyears.sort() 
    fieldSortKeys = list(fieldSort.keys())
    fieldSortKeys.remove("STATE")
    fieldSortKeys.remove("COUNTY")
    fieldSortKeys.sort()
    with open("influenza_data_by_year_by_county.csv", "w") as wp: 
        writerObject = csv.writer(wp, delimiter=',')
        headerRow = ["Year", "County", "Cases", "Population", "Percent"] + fieldSortKeys
        writerObject.writerow(headerRow)
        for year in sortedyears:
            sortedCounties = list(yearSet[year].keys())
            sortedCounties.sort()
            for county in sortedCounties:
                if len(yearSet[year][county]) == 3:
                    targetYear = int(year[year.find("-")+1:])
                    towrite = [targetYear, county] + ['{0:.10f}'.format(x) for x in yearSet[year][county]]
                    if targetYear in yearSet1:
                        if county in yearSet1[targetYear]:
                            for key in fieldSortKeys: 
                                formattedValue = '{0:.10f}'.format(float(yearSet1[targetYear][county][key]))
                                towrite.append(formattedValue)
                    if len(towrite) == (5+len(fieldSortKeys)):
                        writerObject.writerow(towrite)

            
    
    
    
    