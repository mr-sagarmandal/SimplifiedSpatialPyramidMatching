function readDir(path)
% This function illustrates how one can interate through multiple files
% in a directory quickly
% 'path' should be the directory of the files to be read.

files = dir(path); % 'files' is a strcuture storing information
                   % about the files in the directory.

for i=1:length(files)
    disp(files(i).name); % this will display the name of all the files in the directory
    % or, if these files are images
    I = imread(files(i).name);
end


end