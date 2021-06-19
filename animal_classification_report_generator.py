import csv
class Animal_Classification_Report:
    def __init__(self,csv_filename):
        self.csv_filename = csv_filename

    
    def write_csv_header(self,image_name,original_image_path,generated_image_path,prediction,confidence):
        self.image_name = image_name
        self.original_image_path = original_image_path
        self.generated_image_path = generated_image_path
        self.prediction = prediction
        self.confidence = confidence
        f = open(self.csv_filename, "w+",newline='')
        writer = csv.DictWriter(f, fieldnames=[self.image_name,self.original_image_path,self.generated_image_path,self.prediction,self.confidence])
        writer.writeheader()
        f.close()

    def append_csv_rows(self,records):
        self.records = records
        with open(self.csv_filename, 'a+',newline='') as f_object: 
            # Pass this file object to csv.writer() 
            # and get a writer object 
            writer_object = csv.writer(f_object) 
        
            # Pass the list as an argument into 
            # the writerow() 
            writer_object.writerow(self.records) 
        
            #Close the file object 
            f_object.close() 
    
    

