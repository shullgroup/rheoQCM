figure(2);clf(figure(2));plot(freq_data,conductance,'linewidth',3);hold on
[peak_detect,index]=findpeaks(conductance,'minpeakprominence',1e-3,'minpeakheight',(max(conductance)-min(conductance))*.3+min(conductance));
index(1:3)
plot(freq_data,fitted_y(:,1),'ro');
length(index)
plot(freq_data(index),conductance(index,1),'gx-')
peak1=lfun4c([parameters(1:4) parameters(5)+parameters(10)],freq_data);
peak2=lfun4c([parameters(6:9) parameters(5)+parameters(10)],freq_data);
plot(freq_data,peak1);
plot(freq_data,peak2);
plot(freq_data(I),ones(length(I),1).*-1)