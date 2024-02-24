import os


cls = input("Input the class you want to count events for: ")
if cls == 's':
    root = '/mnt/lustre/helios-home/gartmann/tar_data/signal/pdk/PDK/'
elif cls == 'b':
    root = '/mnt/lustre/helios-home/gartmann/tar_data/background/atmonu/AtmoNu/'
else:
    print("Class not recognized!")

print('Counting events in ', root)
group_evt_counts = []

for batch in os.listdir(root):
    if batch.isdigit():
        if cls=='s':
            path_to_group_dir = os.path.join(root, batch, f'files_PDK_{int(batch)}_larcv', 'data')
        elif cls=='b':
            path_to_group_dir = os.path.join(root, batch, f'files_AtmoNu_{int(batch)}_larcv', 'data')

        for group in os.listdir(path_to_group_dir):
            local_counter = 0

            # Full path needed!!
            for evt_dir in os.listdir(os.path.join(path_to_group_dir, group, 'data')):
                if evt_dir.startswith('files_'):
                    local_counter += 1

            group_evt_counts.append(local_counter)

total_count = sum(group_evt_counts)
print(f'Total event count: {total_count}')
