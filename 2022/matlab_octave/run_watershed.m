
folder_names = ['D:\waterhed_implementations_comparison\data\balls\214\dist_balls_214', ...
                ];
folder_names_markers = ['D:\waterhed_implementations_comparison\data\balls\214\markers_balls_214', ...
                        ];
saved_prefix = ['balls64', ...
                ];
                    
assert(size(folder_names,1) == size(folder_names_markers,1) && size(folder_names_markers,1) == size(folder_names_markers,1))
log_folder = 'D:\Documents\matlab\log';
log_path = fullfile(log_folder, "log_watershed.txt");
fid = fopen(log_path, 'a');

%log information about system
fprintf("Get system info\n");
if isunix
    [~,cmdout] = system('lscpu');
else
    [~,cmdout] = system('systeminfo');
end
%fprintf(sprintf("System info:\r\n %s\n\n\n", cmdout));
log_str(sprintf("System info:\r\n %s\r\n\r\n", cmdout), fid);
i = 0
   folder_name = folder_names
   fprintf('%d iteration: process folder %s, prefix: %s\n' , i, folder_name, saved_prefix);
   
   %read relief
   log_str(sprintf("Read relief from: %s", folder_name), fid);
   fprintf("\tReading relief image:\n");
   relief = read_3d_image(folder_name);
   if relief == 0
       fprintf("\t\tCant read relief\n");
      log_str(sprintf("Cant read relief from: %s", folder_name), fid);
   end
   folder_names_marker = folder_names_markers
   %read markers
   log_str(sprintf("Read markers from: %s", folder_names_marker), fid);
   fprintf("\tReading markers image:\n");
   markers = read_3d_image(folder_names_marker);
   if markers == 0
      fprintf("\t\tCant read markers\n");
      log_str(sprintf("Cant read markers from: %s", folder_names_marker), fid);
   end
   
   %create relief via min imposing (with reconstruction)
   log_str("Impose min of relief", fid);
   fprintf("\tImpose min of relief\n");
   relief = imimposemin(relief, markers);
   %clear markers
   
   %sleep half of second for release memory
   pause(0.5);
   
   %[userview,systemview] = memory;
   %log_str(sprintf("Memory usage (userview,systemview): %f, %f", userview.MemUsedMATLAB, ...
   %                 systemview.PhysicalMemory.Total-systemview.PhysicalMemory.Available), fid);
   %fprintf(sprintf("\tMemory usage (userview,systemview): %f, %f\n", userview.MemUsedMATLAB, ...
   %                 systemview.PhysicalMemory.Total-systemview.PhysicalMemory.Available));
   
   
   prompt= "State any string for continue ";
   name=input(prompt,'s');
   
   log_str("Watershed start", fid);
   fprintf("\tWatershed start\n");
                
   %set profiler
   %profile clear
   %profile -memory on
   
   tic
   
   watershed_res = watershed(relief, 26);
   
   toc   
   
   %save profiler result
   %p = profile('info');
   %profile_fname = fullfile(log_folder, strcat(saved_prefix(i), 'myprofiledata.mat'));
   %profile_fname_csv = fullfile(log_folder, strcat(saved_prefix(i), 'myprofiledata.csv'));
   %save(profile_fname, '-struct', 'p')
   %writetable(struct2table(p.FunctionTable), profile_fname_csv)
   
   %save result to folder
   log_str("Result saving", fid);
   fprintf("\tResult saving\n");
   result_save_path = fullfile(log_folder, saved_prefix);
   save_result(watershed_res, result_save_path)
   


fprintf(fid, "\r\n\r\n");

log_str("Processing complete", fid);
fprintf("Processing complete\n");

fclose(fid);





