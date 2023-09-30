import tldextract


def remove_duplicates(input_file, input_file_2, output_file):
    unique_lines = set()

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if line not in unique_lines:
                unique_lines.add(line)

    with open(input_file_2, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip().split('\t')
            fqdn = line[0]
            label = line[1]
            temp = tldextract.extract(fqdn)
            subdomain = temp.subdomain
            newline = subdomain + '\t' + label
            if newline not in unique_lines:
                unique_lines.add(newline)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(unique_lines))


input_filename = 'tunnel_dataset.txt'
input_filename_2 = 'tunnel_data.txt'
output_filename = 'tunnel_dataset_fixed.txt'

remove_duplicates(input_filename, input_filename_2, output_filename)

