import React from 'react';
import WebLinkCard from './WebLinkCard';

export default function WebsitesList({search, data}) {

    return (
        <div className="h-auto w-full px-5 sm:px-10 md:px-20 lg:px-30 py-5 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-10 select-none">
            {data.sort((a,b) => a.ranking - b.ranking).filter(website => website.url.includes(search) || website.title.includes(search)).map((website,index) => {
                    return <WebLinkCard website={website} key={index} />
                })
            }
        </div>
    );
}
