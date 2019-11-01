from igor import binarywave
import h5py

#==========================================
#IBW conversion

def correct_label(label):
    label = [x for x in label if x]  # Remove the empty lists
    label = label[0]  # Remove the unnecessary inception

    corrected_label = []

    for i in label:
        i = i.decode('UTF-8')
        if len(i) == 0:  # Skip empty channel names
            pass
        else:  # Correct the duplicate letters
            if 'cece' in i:
                corrected_label.append(i.replace('cece', 'ce'))
            elif 'aceace' in i:
                corrected_label.append(i.replace('aceace', 'ace'))
            else:
                corrected_label.append(i)
    corrected_label = [x.encode() for x in corrected_label]
    return corrected_label    
    
def ibw2hdf5(filename):
    try:
        tmpdata = binarywave.load(filename)['wave']
        note = tmpdata['note']
        label_list = correct_label(tmpdata['labels'])

        fastsize = float(str(note).split('FastScanSize:')[-1].split('\\r')[0])
        slowsize = float(str(note).split('SlowScanSize:')[-1].split('\\r')[0])
        xoffset = float(str(note).split('XOffset:')[-1].split('\\r')[0])
        yoffset = float(str(note).split('YOffset:')[-1].split('\\r')[0])


        with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
            f.create_dataset("type", data=filename.split('.')[-1])
            f.create_dataset("metadata", data=tmpdata['note'])

            datagrp = f.create_group("datas")
            for i, k in enumerate(label_list):
                datagrp.create_dataset(k, data=tmpdata['wData'][:,:,i])

                datagrp[label_list[i]].attrs['name'] = k.decode('utf8')
                datagrp[label_list[i]].attrs['shape'] = tmpdata['wData'][:,:,i].shape
                datagrp[label_list[i]].attrs['size'] = (fastsize,slowsize)
                datagrp[label_list[i]].attrs['offset'] = (xoffset,yoffset)
                if "Phase" in str(k):
                    datagrp[label_list[i]].attrs['unit'] = ('m', 'm', 'deg')
                elif "Amplitude" in str(k):
                    datagrp[label_list[i]].attrs['unit'] = ('m', 'm', 'V')
                elif "Height" in str(k):
                    datagrp[label_list[i]].attrs['unit'] = ('m', 'm', 'm')
                else:
                    datagrp[label_list[i]].attrs['unit'] = ('m', 'm', 'unknown')


            #f.create_dataset("channelsdata/pxs", data=sizes)

        print('file successfully converted')
    except:
        print('Conversion from .ibw to .hdf5 failed.')