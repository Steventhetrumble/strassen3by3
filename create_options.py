import tables
import numpy as np
import os.path
import json

def check_and_write(array, filename, multiplications, dimension):
    if not os.path.isfile(filename):
        f = tables.open_file(filename, mode='w')
        atom = tables.Float64Atom()
        array_c = f.create_earray(f.root, 'data', atom, (0, dimension**3))
        print (array)
        array_c.append(array)
        f.close()
        print ("new file")
        return True
    f = tables.open_file(filename, mode='r')
    i = 0
    check = 0
    count = 0
    for arr in f.root.data:
        if(count % multiplications == 0):
            c = np.array((f.root.data[count:count+multiplications,0:]))
            idx = np.where(abs((c[:,np.newaxis,:] - array)).sum(axis=2) == 0)
            if (len(idx[0]) == multiplications):
                check += 1
                
        count += 1
    f.close()
    if check > 0:
        print ("Duplicate Solution")
        print(check)
        return False , check
    else:
        print ("Unique Solution")
        f = tables.open_file(filename, mode='a')
        f.root.data.append(array)
        f.close()
        return True , 0


def create_dictionary(options, filename,dimension):
    print("*******************************************")
    print(len(options))
    print("*******************************************")
    f = tables.open_file(filename, mode='w')
    
    for option in options:
        key1 = ""
        for entry in option:
            key1 += str(entry)
        for option2 in options:
            key2 = ""
            for entry2 in option2:
                key2 += str(entry2)
            
            
            
            
            tempArray = []
            for i in range(len(option)):
                for j in range(len(option2)):
                    result = option[i]*option2[j]
                    tempArray.append(result)
            array_c = f.create_array(f.root, key1+key2,tempArray)  
            print(array_c)

    f.close()
    


def find_options(matrix_size, mirror):
    options_size = matrix_size ** 2
    options = []
    if mirror:
        for i in range(int((3 ** options_size))):
            rows = []
            for j in range(0, options_size):
                rows.append(1 - int(i % 3 ** (options_size - j) / (3 ** (options_size - (j + 1)))))
            number = np.count_nonzero(rows)
            if 0 < number < options_size + 1:
                options.append(rows)
        return options
    else:
        for i in range(int((3 ** options_size) / 2)):
            rows = []
            for j in range(0, options_size):
                rows.append(1 - int(i % 3 ** (options_size - j) / (3 ** (options_size - (j + 1)))))
            number = np.count_nonzero(rows)
            if 0 < number < options_size + 1:
                options.append(rows)
        return options


if __name__ == '__main__':
    options = find_options(3, True)

    print(options[len(options) - 1])

    options = find_options(3, False)

    print(options[len(options) - 1])
    # f = tables.open_file("2by2.h5", mode='r')

    # count = 0
    # temp1 = []
    # temp2 = []
    # for item in f.root.data:
    #     if(count % 7 == 0 and count != 0):
    #         temp2.append(temp1)
    #         temp1 = []            
    #     temp1.append(item)

    #     count+=1


    # f.close()
    # total_duplicates = 0
    # temp2 = np.array(temp2)
    # newCount = 0
    # for item in temp2:
    #     print("Solution Number: ",  newCount)
    #     print(item)
    #     result, duplicates = check_and_write(item,"2by2.h5", 7, newCount)
    #     if(result):
    #         break
    #     total_duplicates = duplicates -1
    #     print("total duplicates so far = ", total_duplicates)
    #     newCount += 1
   
        
    

    # for i in range(10):
    #     print(f.root.data[i])
        
        
 
 
 

    # 
    # possible_options = find_options(2, True)
    # dictionary = create_dictionary(possible_options)
    # data = json.dumps(dictionary)
    # with open('2by2data.json', 'w') as outfile:
    #     outfile.write(data)
    # print (dictionary["10-10"]["10-10"])
    # print (type(data))